"""
Need 3 main components:
- Model State
- Actuators
- Update Equations

------------------------------------------
MODEL STATE
- x-pos of vehicle (x)
- y-pos of vehicle (y)
- velocity of vehicle (v)
- yaw (z direction angular velocity)

ACTUATORS
- delta steering angle (delta-sa)
- acceleration (a)

UPDATE EQUATIONS
[
    x(t+1) = x(t) + v(t) * cos(sa) * dt
    y(t+1) = y(t) + v(t) * sin(sa) * dt
    sa(t+1) = sa(t) + (v(t) / Lf) * delta-sa * dt
    v(t+1) = v(t) + a * dt
]
------------------------------------------
INITIAL CONDITIONS
- v = 0
- sa = 0
- x = ?
- y = ?
- theta = ?
------------------------------------------
"""
from .parameters import *

import cvxpy as cp
import numpy as np

class Controller:
    def __init__(self, initial_x, initial_y, ref_states, ref_steer, FPS, verbose=False):
        self.pos_x = initial_x
        self.pos_y = initial_y
        self.ref_states = ref_states
        self.ref_steer = ref_steer
        self.verbose = verbose
        self.dt = np.float32(1 / FPS)
        self.time_horizon = np.linspace(0, TIME_HORIZON)
        self.state = np.zeros([NUM_STATE_VARS])

    def update(self):
        pass

    def iterate(self):
        pass

    def compute(self):
        """
        MPC to compute state and input values for TIME_HORIZON

        :return: array values over the TIME_HORIZON for:
                STATES: x_pos, y_pos, angular_velocity, abs_velocity
                INPUTS: acceleration, steering_angle
        """
        states = cp.Variable((NUM_STATE_VARS, TIME_HORIZON + 1))
        inputs = cp.Variable((NUM_OUTPUTS, TIME_HORIZON))
        cost = 0.0
        constraints = []

        for t in self.time_horizon:
            cost += cp.quad_form(inputs[:, t], I_COST)

            if t != 0:
                cost += cp.quad_form(self.plan[:, t] - states[:, t], S_COST) # Dont know what this means

            A_dt, B_dt, C_dt = self._model_matrix(self.ref_states[2, t],
                                                  self.ref_states[3, t],
                                                  self.ref_steer[0, t])

            constraints += [states[:, t + 1] == A_dt * states[:, t] +
                                                B_dt * inputs[:, t] +
                                                C_dt]

            if t < (TIME_HORIZON - 1):
                # Penalise large input changes
                cost += cp.quad_form(inputs[:, t + 1] - inputs[:, t], I_COST_DIFF) # Penalise large changes
                # Constrain steering angle changes (30 deg/s)
                constraints += [cp.abs(inputs[1, t + 1] - inputs[1, t]) <= MAX_DSTEER * self.dt]

            # Penalise deviation from intended directory
            cost += cp.quad_form(self.ref_states[:, TIME_HORIZON] - states[:, TIME_HORIZON], S_FINAL)

            constraints += [states[:, 0] == 0]
            constraints += [states[2, :] >= 0]
            constraints += [cp.abs(inputs[0, :]) <= 1] # abs(acc or brake) <= 1
            constraints += [cp.abs(inputs[1, :]) <= 0.4] # abs(steering) <= 0.4 rads

            prob = cp.Problem(cp.Minimize(cost), constraints)
            prob.solve(solver=cp.ECOS, verbose=self.verbose)

            # if optimal,  optimal_inaccurate get values from states
            # else return 0s

    def _model_matrix(self, v, phi, delta):
        A_dt = np.eye(NUM_STATE_VARS, dtype=np.float32)
        A_dt[0, 2] = self.dt * np.cos(phi)
        A_dt[1, 2] = self.dt * np.sin(phi)
        A_dt[0, 3] = self.dt * v * np.sin(phi)
        A_dt[1, 3] = self.dt * v * np.cos(phi)
        A_dt[3, 2] = self.dt * np.tan(delta) / CAR_LENGTH

        B_dt = np.zeros((NUM_STATE_VARS, NUM_OUTPUTS))
        B_dt[2, 0] = self.dt
        B_dt[3, 1] = self.dt * v / (CAR_LENGTH * np.cos(delta) ** 2)

        C_dt = np.zeros(NUM_STATE_VARS)
        C_dt[0] = self.dt * v * np.sin(phi) * phi
        C_dt[1] = - self.dt * v * np.cos(phi) * phi
        C_dt[3] = v * delta / (CAR_LENGTH * np.cos(delta) ** 2)

        return A_dt, B_dt, C_dt

