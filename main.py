import numpy as np
from simulation.environment import Environment
from simulation.parameters import FPS
from workspace.MPC import mpc
from utils import c_splines

from pprint import pprint

import matplotlib.pyplot as plt


if __name__ == "__main__":
    env = Environment()
    env.reset()

    ref_states = c_splines.get()
    initial_x, initial_y = env.car.hull.position
    mpc = mpc.Controller(initial_x, initial_y, ref_states, FPS)
    inputs, states= mpc._compute()
    prediction = mpc._predict(mpc.init_states, inputs)
    x = np.arange(prediction.shape[1])

    plt.subplot(411)
    plt.plot(x, prediction[0][:])
    plt.subplot(412)
    plt.plot(x, prediction[1][:])
    plt.subplot(413)
    plt.plot(x, prediction[2][:])
    plt.subplot(414)
    plt.plot(x, prediction[3][:])
    plt.savefig('test.png')


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
