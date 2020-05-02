import numpy as np
from simulation.environment import Environment
from simulation.parameters import FPS
from workspace.MPC import mpc
from utils import c_splines

from pprint import pprint

import matplotlib.pyplot as plt


if __name__ == "__main__":
    env = Environment()
    action = (0, 0, 0)
    isopen = True

    """
        At the first step:
            - get initial state
            - initialise MPC
            - set target index (calc nearest index)
            
        At each step need to:
            - calculate reference trajectory, target state, reference steering?
            - calculate current state 
            - calculate MPC equation
            - update state
    """

    while isopen:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        traj = c_splines.get()
        # Get initial state
        initial_x, initial_y = env.car.hull.position
        # Initialise MPC
        mpc = mpc.Controller(initial_x, initial_y, traj, FPS)

        while True:
            x_pos, y_pos = env.car.hull.position
            # NEED TO BE ABLE TO CACLULATE THESE!!!
            abs_velocity = env.car.hull.
            ang_velocity = env.car.
            state = np.array([x_pos, y_pos,  ])
            actions = mpc.iterate()
            s, r, done, info = env.step(a)
            total_reward += r

            steps += 1

            env.render()
            if done or restart or isopen == False:
                break

    env.close()
