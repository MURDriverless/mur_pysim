from world.parameters import PLAYFIELD


class Rewarder:
    @staticmethod
    def reward(env):
        step_reward = 0

        # Firstly penalise if car is out of bounds
        x, y = env.car.hull.position
        if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
            done = True
            step_reward = -100

        # Since we are assuming car to have infinite fuel, always set fuel_spent to 0
        # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
        env.car.fuel_spent = 0.0

        # Penalty for stopping and wasting time
        env.reward -= 0.1

        # Compute step reward and update previous reward
        step_reward += env.reward - env.prev_reward  # Current recorded reward minus previous reward
        env.prev_reward = env.reward

        return step_reward
