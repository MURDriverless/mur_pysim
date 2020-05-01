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
    def __init__(self, initial_x, initial_y, traj, FPS, verbose=False):
        self.dt = np.float32(1 / FPS)
        self.init_states = np.array([initial_x, initial_y, 2, 0])
        self.traj = traj
        self.ref_steer = 0
        self.sub_ref_states, self.sub_ref_inputs, self.target_idx = \
            self._calc_ref_trajectory(self.init_states, 0, 1, 0)
        self.verbose = verbose
        self.state = np.zeros([NUM_STATE_VARS])

    def update(self, state, a, delta):
        state[0] = state[0] + state[3] * np.cos(state[2]) * self.dt
        state[1] = state[1] + state[3] * np.cos(state[2]) * self.dt
        state[2] = state[2] + state[3] / CAR_LENGTH * np.tan(delta) * self.dt
        state[3] = state[3] + a * self.dt

        return state

    def iterate(self, curr_state):
        pass

    def _compute(self):
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

            A_dt, B_dt, C_dt = self._model_matrix(self.sub_ref_states[2, t],
                                                  self.sub_ref_states[3, t],
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
            cost += cp.quad_form(self.sub_ref_states[:, TIME_HORIZON] - states[:, TIME_HORIZON], S_FINAL)

            constraints += [states[:, 0] == self.init_states]
            constraints += [states[2, :] >= 0]
            constraints += [cp.abs(inputs[0, :]) <= 1] # abs(acc or brake) <= 1
            constraints += [cp.abs(inputs[1, :]) <= 0.4] # abs(steering) <= 0.4 rads

            prob = cp.Problem(cp.Minimize(cost), constraints)
            prob.solve(solver=cp.ECOS, verbose=self.verbose)
            
            # if optimal or optimal_inaccurate get values from states
            if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
                o_states = np.array([
                    np.array(states.value[0, :]).flatten(),
                    np.array(states.value[1, :]).flatten(),
                    np.array(states.value[2, :]).flatten(),
                    np.array(states.value[3, :]).flatten(),
                ])
                o_inputs = np.array([
                    np.array(inputs.value[0, :]).flatten(),
                    np.array(inputs.value[1, :]).flatten()
                ])
            else:
                o_states = np.zeros(4)
                o_inputs = np.zeros(2)

        return o_inputs, o_states

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

    def _calc_ref_trajectory(self, curr_state, curr_idx, curr_step, target_idx):
        # Given the current state, this function determines the predicted states within the time horizon
        ref_state = np.zeros((NUM_STATE_VARS, TIME_HORIZON + 1))
        ref_input = np.zeros((1, TIME_HORIZON + 1))
        course_len = self.traj.shape[1]
        speed_profile = TARGET_SPEED * np.ones(self.traj.shape[1])

        # set initial ref_state values
        dist_travelled = 0.0

        if target_idx >= curr_idx:
            curr_idx = target_idx
        else:
            curr_idx, _ = self._calc_nearest_idx(curr_state, target_idx)

        for i in range(TIME_HORIZON + 1):
            dist_travelled += curr_state[2] * self.dt
            dist_idx = int(dist_travelled / curr_step)

            if (curr_idx + dist_idx) < course_len:
                ref_state[0, i] = self.traj[0][curr_idx + dist_idx]
                ref_state[1, i] = self.traj[1][curr_idx + dist_idx]
                ref_state[2, i] = speed_profile[curr_idx + dist_idx]
                ref_state[3, i] = self.traj[2][curr_idx + dist_idx]
                ref_input[0, i] = 0
            else:
                ref_state[0, i] = self.traj[0][course_len - 1]
                ref_state[1, i] = self.traj[1][course_len - 1]
                ref_state[2, i] = speed_profile[course_len - 1]
                ref_state[3, i] = self.traj[2][course_len - 1]
                ref_input[0, i] = 0

        return ref_state, ref_input, curr_idx

    def _calc_nearest_idx(self, curr_state, target_idx):
        dx = [curr_state[0] - i_traj for i_traj in self.traj[0][target_idx:(target_idx + N_IDX_SEARCH)]]
        dy = [curr_state[1] - i_traj for i_traj in self.traj[1][target_idx:(target_idx + N_IDX_SEARCH)]]

        dist = [i_dx ** 2 + i_dx ** 2 for i_dx, i_dy in zip(dx, dy)]
        dist_min = min(dist)

        target = dist.index(dist_min) + target_idx
        dist_min = np.sqrt(dist_min)

        return target, dist_min








