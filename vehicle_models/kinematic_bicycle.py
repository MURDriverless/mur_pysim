import math
import numpy as np

# There are 4 states and 2 inputs
NX = 4  # x, y, v, yaw
NU = 2  # acceleration (a) and steering (delta or d)
L = 2.5

# Constraints of the kinematic bicycle plant
MAX_STEER = np.deg2rad(45.0)  # maximum steering angle [rad]
MAX_DSTEER = np.deg2rad(30.0)  # maximum steering speed [rad/s]
MAX_SPEED = 55.0 / 3.6  # maximum speed [m/s]
MIN_SPEED = -20.0 / 3.6  # minimum speed [m/s]
MAX_ACCEL = 1.0  # maximum accel [m/ss]
MIN_ACCEL = -1.0  # maximum braking [m/ss]


def nonlinear_model_c(x, u):
    """
    Non-linear continuous kinematics of Kinematic Bicycle

    Args:
        x (np.ndarray): 1D vector of size NX
        u (np.ndarray): 1D vector of size NU

    Returns:
        np.ndarray: dxdt, which is the derivative of the states
    """
    dxdt = np.zeros(NX)
    dxdt[0] = x[2] * math.cos(x[3])  # x_dot
    dxdt[1] = x[2] * math.sin(x[3])  # y_dot
    dxdt[2] = u[0]  # v_dot (acceleration)
    dxdt[3] = (x[2] * math.tan(u[1])) / L  # yaw_dot
    return dxdt


def nonlinear_model_d(xk, uk, Ts, repeat_sampling=False):
    """
    Non-linear discrete kinematics of Kinematic Bicycle

    Args:
        xk (np.ndarray): 1D vector of size NX, which are states at timestep k
        uk (np.ndarray): 1D vector of size NU, which are inputs at timestep k
        Ts (float): Sampling rate
        repeat_sampling (bool): If True, approximates xk1 at a smaller timestep

    Returns:
        np.ndarray: xk1, which is the predicted next state
    """
    # Constrain inputs to maximum and minimum values
    # Acceleration
    if uk[0] >= MAX_ACCEL:
        uk[0] = MAX_ACCEL
    elif uk[0] <= MIN_ACCEL:
        uk[0] = MIN_ACCEL
    # Delta
    if uk[1] >= MAX_STEER:
        uk[1] = MAX_STEER
    elif uk[1] <= -MAX_STEER:
        uk[1] = -MAX_STEER

    if repeat_sampling:
        M = 10  # number of repeated samplings
        dt = Ts / M
        xk1 = xk
        for i in range(0, M):
            dxdt = nonlinear_model_c(xk1, uk)
            xk1 = np.add(xk1, dt * dxdt)
    else:
        dt = Ts
        dxdt = nonlinear_model_c(xk, uk)
        xk1 = np.add(xk, dt * dxdt)
    return xk1


def linear_model_d(v, yaw, delta, Ts):
    """
    Obtain A, B, C of linearised discrete kinematics via Euler Forward & Taylor's 1st Order Approximation

    Args:
        v (float): Velocity of centre of vehicle
        yaw (float): Current heading of the vehicle centre
        delta (float): Current steering input into the vehicle
        Ts (float): Controller sampling period

    Returns:
        np.ndarray: [A, B, C]
    """
    # Pre-compute variables to reduce computation
    sin_yaw = math.sin(yaw)
    cos_yaw = math.cos(yaw)
    cos_delta = math.cos(delta)

    # Compute Jacobians of A and B at equilibrium
    A_lin = np.zeros((NX, NX))
    A_lin[0, 2] = cos_yaw
    A_lin[0, 3] = -v * sin_yaw
    A_lin[1, 2] = sin_yaw
    A_lin[1, 3] = v * cos_yaw
    A_lin[3, 2] = math.tan(delta) / L
    B_lin = np.zeros((NX, NU))
    B_lin[2, 0] = 1
    B_lin[3, 1] = v / (L * (cos_delta ** 2))

    # Note that A and B are the returned linearised state difference equations
    A = np.add(np.eye(NX), A_lin * Ts)
    B = B_lin * Ts

    # C is not the observation matrix, but instead a constant term in Taylor's approximation.
    # Refer to Atsushi Sakai's PathTracking/Model_predictive_speed_and_steering_control.ipynb
    C = np.zeros(NX)
    C[0] = Ts * v * sin_yaw * yaw
    C[1] = - Ts * v * cos_yaw * yaw
    C[3] = - Ts * v * delta / (L * (cos_delta ** 2))
    return A, B, C
