import numpy as np


class State:
    def __init__(self, env):
        self.car = env.car
        self.info = {
            "x_pos": self.car.hull.position[0],
            "y_pos": self.car.hull.position[1],
            "abs_vel": np.sqrt(self.car.hull.linearVelocity[0] ** 2 +
                               self.car.hull.linearVelocity[1] ** 2),
            "angular_vel": self.car.hull.angularVelocity,
        }

    def __getitem__(self, item: str) -> float:
        return self.info.get(item)

    def update(self, env) -> dict:
        car = env.car

        abs_vel = np.sqrt(car.hull.linearVelocity[0] ** 2 +
                          car.hull.linearVelocity[1] ** 2)

        self.info.update({
            "x_pos": car.hull.position[0],
            "y_pos": car.hull.position[1],
            "abs_vel": abs_vel,
            "angular_vel": car.hull.angularVelocity,
        })

        return np.array(tuple(self.info.values()))




