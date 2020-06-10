import numpy as np
import cvxpy
from vehicle_models.kinematic_bicycle import KinematicBicycleModel as model
from path_followers.interface import PathFollowerInterface


class LTVMPCFollower(PathFollowerInterface):
    def __init__(self, N, Ts):
        """
        Args:
            N (int): prediction horizon length
            Ts (float): sampling period
        """
        self.N = N
        self.Ts = Ts

        # State, terminal state, input and input difference costs respectively
        self.Q = np.diag([1.0, 1.0, 0, 0.5])
        self.P = self.Q
        self.R = np.diag([0.01, 0.01])
        self.Rd = np.diag([0.01, 1.0])

        # We store past optimal inputs for future linearisations.
        self.pa = np.ones(self.N)
        self.pd = np.zeros(self.N)
        self.x0 = np.zeros(model.NX)
        self.xref = np.zeros((model.NX, self.N + 1))

    def move(self, state, reference):

        x0 = self.x0
        x0[0] = state[0]
        x0[1] = state[1]
        x0[2] = state[2]
        x0[3] = state[3]

        xref = self.xref
        xref[0, :] = reference[0, :]
        xref[1, :] = reference[1, :]
        xref[3, :] = np.arctan2(xref[1, :], xref[2, :])

        # Predict equilibrium values for the future horizons
        xbar = self.calculate_xbar(x0, self.pa, self.pd)
        # Compute optimal inputs
        oa, od = self.ltv_mpc_control(x0, xref, xbar)
        # Update previous inputs for calculating xbar in the next iteration
        self.pa, self.pd = oa, od
        # Return result
        return oa, od

    def calculate_xbar(self, x0, pa, pd):
        """
        Predict state equilibria in the range of [0, N] inclusive, by applying
        the current state and previous optimal inputs to the predictive model.

        Args:
            x0 (numpy.ndarray): 1D vector of size NX containing current state
            pa (numpy.ndarray): 1D vector of size N listing previous acceleration
            pd (numpy.ndarray): 1D vector of size N listing previous delta

        Returns:
            numpy.ndarray: 2D vector of size (NX, N+1)
        """
        # Because we start from horizon 0 and end at N, the length is N+1
        xbar = np.zeros((model.NX, self.N + 1))
        xk = np.zeros(model.NX)
        uk = np.zeros(model.NU)

        # At horizon 0, xbar is just x0. Also, initialise xk and uk using x0
        # and past optimal inputs
        for i in range(len(x0)):
            xbar[i, 0] = x0[i]
            xk[i] = x0[i]
        uk[0] = pa[0]
        uk[1] = pd[0]

        # Calculate xbar from horizon 1 to N, so we have to use N+1 as the end range
        for (ai, di, i) in zip(pa, pd, range(1, self.N + 1)):
            # Predict next position using previous input and current state
            x, y, v, yaw = model.discrete_model(xk, uk, self.Ts)
            # Collect xbar result
            xbar[0, i] = xk[0]
            xbar[1, i] = xk[1]
            xbar[2, i] = xk[2]
            xbar[3, i] = xk[3]
            # Update state for next horizon
            xk[0] = x
            xk[1] = y
            xk[2] = v
            xk[3] = yaw
            # Update input for next horizon
            uk[0] = ai
            uk[1] = di
        return xbar

    def ltv_mpc_control(self, x0, xref, xbar):
        """
        Computes optimal acceleration and delta up to the horizon N

        Args:
            x0 (numpy.ndarray): 1D vector of size NX containing the current state
            xref (numpy.ndarray): 2D vector of size (NX, N+1) containing the reference
            xbar (numpy.ndarray): 2D vector of size (NX, N+1) containing the equilibria at each horizon

        Returns:
            np.ndarray: 2D vector of size (NX, N) -> [oa; ow]
                It lists the optimal acceleration and optimal delta for horizons up to N
        """
        # Initialisation:
        # Because we start from initial state 0 and end at N, the length is N+1
        x = cvxpy.Variable((model.NX, self.N + 1))
        # The terminal state is influenced by u at previous horizon, so we only include u
        # up to horizon N-1
        u = cvxpy.Variable((model.NU, self.N))
        cost = 0.0
        constraints = []
        # ubar for linearised kinematics. The only term in ubar which is used in the Jacobian
        # is dbar (equilibrium of steering angle delta), which we set to 0.0 for now
        ubar = np.zeros(model.NU)
        ubar[1] = 0.0

        # Add stage cost and constraints
        for t in range(self.N):
            # Penalise tracking error
            cost += cvxpy.quad_form(xref[:, t] - x[:, t], self.Q)
            # Constrain next state to follow predictive model
            A, B, C = model.discrete_jacobian(xbar[:, t], ubar, self.Ts)
            constraints += [x[:, t + 1] == A * x[:, t] + B * u[:, t] + C]

            # Maximise velocity
            cost += -x[2, t] * 0.1

            # Penalise input and input difference
            cost += cvxpy.quad_form(u[:, t], self.R)
            # Only penalise input difference up to N-1 to build up a difference
            if t < (self.N - 1):
                cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], self.Rd)
                # Limit the steering change rate
                constraints += [cvxpy.abs(u[1, t + 1] - u[1, t]) <= model.MAX_DSTEER * self.Ts]

        # Add terminal stage cost
        cost += cvxpy.quad_form(xref[:, self.N] - x[:, self.N], self.P)

        # Initial state must be the same as x0
        constraints += [x[:, 0] == x0]
        # Constrain speed
        constraints += [x[2, :] <= model.MAX_SPEED]
        constraints += [x[2, :] >= model.MIN_SPEED]
        # Constrain acceleration
        constraints += [u[0, :] <= model.MAX_ACCEL]
        constraints += [u[0, :] >= model.MIN_ACCEL]
        # Constrain delta
        constraints += [cvxpy.abs(u[1, :]) <= model.MAX_STEER]

        # Solve optimisation problem
        problem = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        problem.solve(solver=cvxpy.ECOS, verbose=False)

        if problem.status == cvxpy.OPTIMAL or problem.status == cvxpy.OPTIMAL_INACCURATE:
            # ox = np.array(x.value[0, :]).flatten()
            # oy = np.array(x.value[1, :]).flatten()
            # ov = np.array(x.value[2, :]).flatten()
            # oyaw = np.array(x.value[3, :]).flatten()
            oa = np.array(u.value[0, :]).flatten()
            od = np.array(u.value[1, :]).flatten()
        else:
            print("No solution")
            ox, oy, ov, oyaw, oa, od = None, None, None, None, None, None
        return oa, od
