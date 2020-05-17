import numpy as np
from simulation.environment import Environment
from workspace.MPC.mpc import Controller
from workspace.MPC.transform_action import transform
from utils.c_splines import get

if __name__ == "__main__":
    env = Environment()
    env.render()
    action = (0, 0, 0)
    isopen = True

    c_splines = get()
    initial_states = np.zeros(4)
    mpc = Controller(initial_states, c_splines)

    while isopen:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            s, r, done, info = env.step(action)
            state = [s['x_pos'],
                     s['y_pos'],
                     s['abs_vel'],
                     s['angular_vel']]

            action = mpc.iterate(state)
            action = transform(action)
            print(action)

            steps += 1
            isopen = env.render()

    env.close()
