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
        c_splines  = c_splines.get()
        # Get initial state
        initial_x, initial_y = env.car.hull.position
        initial_states = np.array([initial_x, initial_y, 0, 0])
        # Initialise MPC
        mpc = mpc.Controller(initial_states, c_splines)
        action = np.zeros(2)

        while True:
            # x, y position
            x_pos, y_pos = env.car.hull.position
            # velocity
            abs_velocity = env.car.velocity
            # steering angle
            sa = env.car.wheels[0].phase
            # state formatting
            current_state = np.array([x_pos, y_pos, abs_velocity, 0])
            # MPC prescription and optimal action identification
            action = mpc.iterate(current_state, sa)
            action = np.hstack((action, 0))
            print(action)

            s, r, done, info = env.step(action)
            total_reward += r
            steps += 1

            env.render()
            if done or restart or isopen == False:
                break

    env.close()
