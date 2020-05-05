import numpy as np

TARGET_SPEED = 5
N_IDX_SEARCH = 10

WHEELPOS = [
    (-55, +80), (+55, +80),
    (-55, -82), (+55, -82)
]

# Vehicle Parameters
CAR_LENGTH = WHEELPOS[0][1] - WHEELPOS[2][1]
CAR_WIDTH = WHEELPOS[1][0] - WHEELPOS[0][0]

# Physics Parameters (Need to source)
# MAX_STEER (rad)
# MAX_STEER_DELTA (rad/s)
# MAX_SPEED (pixels/step)
# MIN_SPEED (pixels/step)
# MAX_ACC (pixels/s/s)

# MPC Parameters
NUM_STATE_VARS = 4
NUM_OUTPUTS = 2
TIME_HORIZON = 5 # steps

MAX_DSTEER = np.radians(15)

I_COST = np.diag([0.01, 0.01])
I_COST_DIFF = np.diag([0.01, 1.0])
S_COST = np.diag([1.0, 1.0, 0.5, 0.5])
S_FINAL = S_COST




