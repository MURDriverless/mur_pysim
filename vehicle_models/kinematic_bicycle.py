import math
import numpy as np
from vehicle_models.interface import VehicleModelInterface


class KinematicBicycleModel(VehicleModelInterface):
    NX = 4      # 4 states: [x, y, v, yaw]
    NU = 2      # 2 inputs: [acceleration, delta (steering)]
    L = 2.5     # Length of vehicle from rear to front

    @classmethod
    def get_NX(cls):
        return cls.NX

    @classmethod
    def get_NU(cls):
        return cls.NU

    @classmethod
    def continuous_model(cls, x, u):
        """
        Non-linear continuous kinematics of the vehicle

        Args:
            x (np.ndarray): 1D vector of size NX
            u (np.ndarray): 1D vector of size NU

        Returns:
            np.ndarray: dxdt, which is the derivative of the states, of size (NX, 1)
        """
        dxdt = np.zeros(cls.NX)
        dxdt[0] = x[2] * math.cos(x[3])  # x_dot
        dxdt[1] = x[2] * math.sin(x[3])  # y_dot
        dxdt[2] = u[0]  # v_dot (acceleration)
        dxdt[3] = (x[2] * math.tan(u[1])) / cls.L  # yaw_dot
        return dxdt

    @classmethod
    def discrete_model(cls, xk, uk, dt):
        """
        Uses cls.continuous_model() to discretise model via Euler Forward method

        Args:
            xk (np.ndarray): 1D vector of size NX, which are states at timestep k
            uk (np.ndarray): 1D vector of size NU, which are inputs at timestep k
            dt (float): Time-step size

        Returns:
            np.ndarray: xk1, which is the predicted next state, of size (NX, 1)
        """
        dxdt = cls.continuous_model(xk, uk)
        xk1 = np.add(xk, dt * dxdt)
        return xk1

    @classmethod
    def discrete_jacobian(cls, xk_bar, uk_bar, dt):
        """
        Obtain A, B, C of linearised discrete kinematics via Euler Forward & Taylor's 1st Order Approximation

        Args:
            xk_bar (np.ndarray): 1D vector of size NX, which are equilibrium states at time-step k
            uk_bar (np.ndarray): 1D vector of size NU, which are equilibrium inputs at time-step k
            dt (float): Time-step size

        Returns:
            np.ndarray: [A, B, C], which fulfils the linearised dynamics | xk1 = A * x[k] + B * u[k] + C | where
                - A: Linear A at equilibrium with respect to state, of size (NX, NX)
                - B: Linear B at equilibrium with respect to state, of size (NX, NU)
                - C: Feed-forward term for the dynamics of size (NX, 1). It should at least return a zero matrix. For
                    more information, refer to Atsushi Sakai's PathTracking/Model_predictive_speed_and_steering_control
        """
        J_A = np.zeros((cls.NX, cls.NX))    # Jacobian of A, same size as linearised A
        J_B = np.zeros((cls.NX, cls.NU))    # Jacobian of B, same size as linearised B
        J_C = np.zeros((cls.NX, 1))         # Collection of terms not in J_A and J_B, same size as feed-forward term C
        L = cls.L
        v = xk_bar[2]
        yaw = xk_bar[3]
        delta = uk_bar[1]

        # Pre-compute terms to reduce computation
        sin_yaw = math.sin(yaw)
        cos_yaw = math.cos(yaw)
        cos_delta = math.cos(delta)

        # Compute J_A, J_B and J_C
        J_A[0, 2] = cos_yaw
        J_A[0, 2] = cos_yaw
        J_A[0, 3] = -v * sin_yaw
        J_A[1, 2] = sin_yaw
        J_A[1, 3] = v * cos_yaw
        J_A[3, 2] = math.tan(delta) / L
        # -----------------------------
        J_B[2, 0] = 1
        J_B[3, 1] = v / (L * (cos_delta ** 2))
        # -----------------------------
        J_C[0] = v * sin_yaw * yaw
        J_C[1] = -v * cos_yaw * yaw
        J_C[3] = -v * delta / (L * (cos_delta ** 2))

        # Assemble A, B and C
        A = np.add(np.eye(cls.NX), J_A * dt)    # A = (I + A' * dt), where A' is J_A
        B = J_B * dt                            # B = B' * dt, where B' is J_B
        C = J_C * dt                            # C = C' * dt, where C' is J_C
        return A, B, C
