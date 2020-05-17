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
        curr_state = env.reset()
        total_reward = 0.0
        steps = 0
        restart = False

        # Generate path
        c_splines = c_splines.get()

        # Initialise MPC
        mpc = mpc.Controller(curr_state, c_splines)

        # Set initial inputs to zero
        action = np.zeros(2)

        while True:
            # x, y position
            # velocity
            # state formatting
            # MPC prescription and optimal action identification
            action = mpc.iterate(curr_state)
            action = np.array([action[1], action[0], 0])

            # 0: acc/brake
            # 1: steering

            if action[1] <= 0:
                action[2] = -action[1]
                action[1] = 0

            curr_state = env.step(action)
            steps += 1

            env.render()

    env.close()
