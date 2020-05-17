import numpy as np


def transform(raw_action):
    action = np.zeros(3)

    if raw_action[0] > 0:
        action[1] = raw_action[0]
    elif raw_action[0] < 0:
        action[2] = abs(raw_action[0])

    action[0] = raw_action[1]

    return action
