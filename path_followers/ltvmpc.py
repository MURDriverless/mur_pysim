import numpy as np
import cvxpy
from path_followers.contract import PathFollowerContract
from vehicle_models.kinematic_bicycle import NX, NU, MAX_STEER, MAX_DSTEER, MAX_SPEED, MIN_SPEED, \
    MAX_ACCEL, MIN_ACCEL, nonlinear_model_d, linear_model_d


class LTVMPCFollower(PathFollowerContract):
    def __init__(self):
        self.Ts = 0.2
        self.horizon_length = 5

    def move(self, states, reference=None):
        x0 = states

    def calculate_xbar(self, x0, oa, od):
        """
        Computes xbar for the range [0, horizon_length] (inclusive), by applying
        optimal acceleration and delta at each iteration

        Args:
            x0 (numpy.ndarray): 1D vector of size NX containing current state
            oa (numpy.ndarray): 1D vector of size horizon_length listing optimal acceleration
            od (numpy.ndarray): 1D vector of size horizon_length listing optimal delta

        Returns:
            numpy.ndarray: 2D vector of size (NX, horizon_length + 1)
        """
        # Reference the variables in "self"
        Ts = self.Ts
        horizon_length = self.horizon_length

        # Initialise variables
        xbar = np.zeros((NX, horizon_length + 1))
        xk = np.zeros(NX)
        uk = np.zeros(NU)
        uk[0] = oa[0]
        uk[1] = od[0]

        # At horizon = 0, xbar is the current state, which is x0
        for i in range(len(x0)):
            xbar[i, 0] = x0[i]
            xk[i] = x0[i]

        # In the next horizons, calculate xbar by applying optimal oa and od to the nonlinear discrete model
        # Note that we set end range to be "horizon_length + 1" as we are iterating from 1.
        # This is because the first horizon is the current measurements
        for (ai, di, i) in zip(oa, od, range(1, horizon_length + 1)):
            # Apply input to plant
            x, y, v, yaw = nonlinear_model_d(xk, uk, Ts)
            # Update state from plant
            xk[0] = x
            xk[1] = y
            xk[2] = v
            xk[3] = yaw
            # Update optimal inputs for next iteration
            uk[0] = ai
            uk[1] = di
            # Set xbar for subsequent horizons to be exactly the states from applying
            # optimal inputs
            xbar[0, i] = xk[0]
            xbar[1, i] = xk[1]
            xbar[2, i] = xk[2]
            xbar[3, i] = xk[3]
        return xbar

    def ltv_mpc_control(self, x0, xref, dref, xbar):
        """
        Computes optimal acceleration and delta for subsequent horizons, given the current state, states reference,
        steering input reference and the equilibrium value xbar to linearise the car kinematics

        Args:
            x0 (numpy.ndarray): 1D vector of size NX containing the current state
            xref (numpy.ndarray): 2D vector of size (NX, horizon_length + 1) containing the current state reference
            dref (numpy.ndarray): 2D vector of size (1, horizon_length) containing the current steering reference
            xbar (numpy.ndarray): 2D vector of size (NX, horizon_length+1) containing the equilibrium values at each horizon

        Returns:
            np.ndarray: 2D vector of size (NX, horizon_length) -> [oa; ow]
                It lists the optimal acceleration and optimal delta for horizons up to horizon_length
        """
        Ts = self.Ts
        horizon_length = self.horizon_length

        Q = np.diag([1.0, 1.0, 0.5, 0.5])  # state cost matrix
        Qf = Q  # final state cost matrix
        R = np.diag([0.01, 0.01])  # input cost matrix
        Rd = np.diag([0.01, 1.0])  # input difference cost matrix

        x = cvxpy.Variable((NX, horizon_length + 1))
        u = cvxpy.Variable((NU, horizon_length))
        cost = 0.0
        constraints = []

        for t in range(horizon_length):
            cost += cvxpy.quad_form(u[:, t], R)

            # Penalise state deviation from reference
            if t != 0:
                cost += cvxpy.quad_form(xref[:, t] - x[:, t], Q)

            A, B, C = linear_model_d(xbar[2, t], xbar[3, t], dref[0, t], Ts)
            constraints += [x[:, t + 1] == A * x[:, t] + B * u[:, t] + C]

            if t < (horizon_length - 1):
                cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], Rd)
                # Saturate the changes in steering from previous state
                constraints += [cvxpy.abs(u[1, t + 1] - u[1, t]) <= MAX_DSTEER * Ts]

        # At the end of the horizon, add cost of Qf
        cost += cvxpy.quad_form(xref[:, horizon_length] - x[:, horizon_length], Qf)

        # Initial state must be the same as x0
        constraints += [x[:, 0] == x0]
        # Constrain speed
        constraints += [x[2, :] <= MAX_SPEED]
        constraints += [x[2, :] >= MIN_SPEED]
        # Constrain acceleration
        constraints += [u[0, :] <= MAX_ACCEL]
        constraints += [u[0, :] >= MIN_ACCEL]
        # Constrain delta
        constraints += [cvxpy.abs(u[1, :]) <= MAX_STEER]

        # Solve optimisation problem
        problem = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        problem.solve(solver=cvxpy.ECOS, verbose=False)

        if problem.status == cvxpy.OPTIMAL or problem.status == cvxpy.OPTIMAL_INACCURATE:
            ox = np.array(x.value[0, :]).flatten()
            oy = np.array(x.value[1, :]).flatten()
            ov = np.array(x.value[2, :]).flatten()
            oyaw = np.array(x.value[3, :]).flatten()
            oa = np.array(u.value[0, :]).flatten()
            od = np.array(u.value[1, :]).flatten()
        else:
            print("No solution")
            ox, oy, ov, oyaw, oa, od = None, None, None, None, None, None
        return ox, oy, ov, oyaw, oa, od

