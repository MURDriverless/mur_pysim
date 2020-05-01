import numpy as np
from simulation.environment import Environment
from simulation.parameters import FPS
from workspace.MPC import mpc
from utils import c_splines


if __name__ == "__main__":
    env = Environment()
    env.reset()

    ref_states = c_splines.get()
    initial_x, initial_y = env.car.hull.position
    mpc = mpc.Controller(initial_x, initial_y, ref_states, FPS)


    action = (0, 0, 0)

    isopen = True

    """
    while isopen:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            s, r, done, info = env.step(a)
            total_reward += r
            if steps % 200 == 0 or done:
                print(s)
                print("action " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            steps += 1
            isopen = env.render()
            if done or restart or isopen == False:
                break

    env.close()
    """
