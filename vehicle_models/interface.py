from abc import ABCMeta, abstractmethod


class VehicleModelInterface(metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def continuous_model(cls, x, u):
        """
        Non-linear continuous dynamics of the vehicle

        Args:
            x (np.ndarray): 1D vector of size NX
            u (np.ndarray): 1D vector of size NU

        Returns:
            np.ndarray: dxdt, which is the derivative of the states, of size (NX, 1)
        """
        pass

    @classmethod
    @abstractmethod
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
        pass

    @classmethod
    @abstractmethod
    def discrete_jacobian(cls, xk_bar, uk_bar, dt):
        """
        Obtain A, B, C of linearised discrete model via Euler Forward & Taylor's 1st Order Approximation

        Args:
            xk_bar (np.ndarray): 1D vector of size NX, which are equilibrium states at time-step k
            uk_bar (np.ndarray): 1D vector of size NU, which are equilibrium inputs at time-step k
            dt (float): Time-step size

        Returns:
            np.ndarray: [A, B, C], which fulfils the linearised dynamics | xk1 = A * x[k] + B * u[k] + C | where
                - A: Linear A at equilibrium with respect to state, of size (NX, NX)
                - B: Linear B at equilibrium with respect to state, of size (NX, NU)
                - C: Feed-forward term for the dynamics of size (NX, 1). It should at least return a zero matrix
        """
        pass
