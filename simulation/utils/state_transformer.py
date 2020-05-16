import numpy as np

class StateTransformer:
    @staticmethod
    def transform(env):
        """
        Parameters
        ----------
        env: Environment

        Returns
        -------
        dict
        It should contain the following:
        - x, y of car
        """
        car = env.car
        x, y = car.hull.position

        return {
            'x': x,
            'y': y,
            'velocity': np.sqrt(car.LinearVelocity[0] ** 2 + car.LinearVelocity[1] ** 2)
        }
