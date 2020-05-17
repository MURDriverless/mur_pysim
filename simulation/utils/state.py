import numpy as np


class State:
    def __init__(self, env):
        self.env = env
        self.car = self.env.car
        self.info = {
            "x_pos": self.car.hull.position[0],
            "y_pos": self.car.hull.position[1],
            "x_vel": self.car.hull.linearVelocity[0],
            "y_vel": self.car.hull.linearVelocity[1],
            "angular_vel": self.car.hull.angularVelocity
            }

        self.time = self.env.time

    def __getitem__(self, item: str) -> float:
        return self.info.get(item)

    def update(self, env) -> dict:
        car = env.car

        abs_vel = np.sqrt(car.hull.linearVelocity[0] ** 2 +
                          car.hull.linearVelocity[1] ** 2)

        self.info.update({
            "x_pos": car.hull.position[0],
            "y_pos": car.hull.position[1],
            "x_vel": car.hull.linearVelocity[0],
            "y_vel": car.hull.linearVelocity[1],
            "abs_vel": abs_vel,
            "angular_vel": car.hull.angularVelocity,
            "steering_ang": car.wheels[0].joint.angle
        })

