from world.parameters import FPS


class PhysicsEngine:
    def __init__(self, env):
        self.env = env

    def step(self, action):
        car = self.env.car
        world = self.env.world
        time = self.env.time

        if action is not None:
            car.steer(-action[0])
            car.gas(action[1])
            car.brake(action[2])

        car.step(1.0 / FPS)
        world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        time += 1.0 / FPS
