import numpy as np

TARGET_SPEED = 20.0
N_IDX_SEARCH = 15

WHEELPOS = [
    (-55, +80), (+55, +80),
    (-55, -82), (+55, -82)
]

# Vehicle Parameters
CAR_LENGTH = (WHEELPOS[0][1] - WHEELPOS[2][1]) / 2
CAR_WIDTH = WHEELPOS[1][0] - WHEELPOS[0][0]

# Physics Parameters (Need to source)
# MAX_STEER (rad)
# MAX_STEER_DELTA (rad/s)
# MAX_SPEED (m/s)
# MIN_SPEED (m/s)
# MAX_ACC (m/s/s)

# MPC Parameters
NUM_STATE_VARS = 4
NUM_OUTPUTS = 2
TIME_HORIZON = 20 # steps

MAX_DSTEER = np.radians(5)

I_COST = np.diag([0.01, 0.5])
I_COST_DIFF = np.diag([0.01, 0.9])
S_COST = np.diag([0.75, 0.75, 0.05, 0.05])
S_FINAL = S_COST




