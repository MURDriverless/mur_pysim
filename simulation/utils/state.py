import numpy as np


class State:
    def __init__(self, env):
        self.car = env.car
        self.info = {
            "x_pos": self.car.hull.position[0],
            "y_pos": self.car.hull.position[1],
            "abs_vel": np.hypot(self.car.hull.linearVelocity[0],
                               self.car.hull.linearVelocity[1]),
            "angular_vel": self.car.hull.angularVelocity,
        }

    def __getitem__(self, item: str) -> float:
        return self.info.get(item)

    def update(self, env) -> dict:
        car = env.car
        self.info.update({
            "x_pos": car.hull.position[0],
            "y_pos": car.hull.position[1],
            "abs_vel": np.hypot(car.hull.linearVelocity[0],
                                car.hull.linearVelocity[1]),
            "angular_vel": car.hull.angularVelocity
        })

        return np.array(tuple(self.info.values()))

    def calc_velocity(self):







