"""
    2019 YP2 MPC FORMULATION

    1. Set initial inputs to 0
    2. Initialise inputs (k - 1) using increasing horizon initialisation
    3. Set time step = 0

    While TIMESTEP < END TIMESTEP do:
        4. Obtain guesses for inputs over horizon n
        5. Calculate guesses for states over horizon n
        6. Linearise system dynamics over GUESSES (A, B, C)
        7. Linearise track boundary orientation function over guesses ????
        8. Linearise track error over guesses
        9. Solve LTV MPC with virtual state over horizon n
        10. Calculate inputs over horizon n handling constraints
        11. Set n = n + 1
"""
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

        # Generate path
        c_splines = c_splines.get()

        # Get initial state
        initial_x, initial_y = env.car.hull.position
        initial_states = np.array([initial_x, initial_y, 0, 0])

        # Initialise MPC
        mpc = mpc.Controller(initial_states, c_splines)

        # Set initial inputs to zero
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
            if action[1] <= 0:
                action[2] = -action[1]
                action[1] = 0


            print(action)

            s, r, done, info = env.step(action)
            total_reward += r
            steps += 1

            env.render()
            if done or restart or isopen == False:
                break

    env.close()
