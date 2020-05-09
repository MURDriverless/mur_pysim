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
import sys
sys.path.append("../..")
from simulation.parameters import FPS
from time import time

import cvxpy as cp
import numpy as np

DT = np.float32(5 / FPS)


class Controller:
    """
        States are:
            0: x-pos of vehicle
            1: y-pos of vehicle
            2: absolute velocity of vehicle
            3: yaw (z direction angular velocity)

        Inputs are:
            0: acc/brake
            1: steering
    """
    def __init__(self, i_states, c_splines, verbose=False):
        self.i_states = i_states
        self.c_splines = c_splines
        self.state = np.zeros([NUM_STATE_VARS])
        self.steps = 1
        self.index = 0
        self.verbose = verbose
        self.o_inputs = np.zeros((2, TIME_HORIZON), dtype=np.float32)
        self.o_input = np.zeros(2, dtype=np.float32)
        self.spline_dist = self._calc_spline_dist()

    def iterate(self, curr_state, delta):
        if self.steps == 1:
            ref_states, ref_inputs = self._calc_ref_traj(self.i_states)
            predict_states = self._predict(self.i_states)
            o_inputs, o_states = self._compute(ref_states, ref_inputs, predict_states, curr_state)

        else:
            curr_state[3] = self._calc_yaw(curr_state, delta)
            ref_states, ref_inputs = self._calc_ref_traj(curr_state)
            predict_states = self._predict(curr_state)
            o_inputs, o_states = self._compute(ref_states, ref_inputs, predict_states, curr_state)

        self.steps += 1
        return o_inputs[:, 0]

    def _calc_spline_dist(self):
        x = (self.c_splines[0][0] - self.c_splines[0][1]) ** 2
        y = (self.c_splines[1][0] - self.c_splines[1][1]) ** 2

        return np.sqrt(x + y)

    @staticmethod
    def _calc_yaw(curr_state, delta):
        return (curr_state[2] * np.tan(delta)) / CAR_LENGTH

    def _calc_ref_traj(self, curr_state):
        # 0.004 seconds
        ref_state = np.zeros((NUM_STATE_VARS, TIME_HORIZON + 1))
        ref_input = np.zeros((NUM_OUTPUTS, TIME_HORIZON + 1))
        course_len = self.c_splines.shape[1]
        speed_profile = TARGET_SPEED * np.ones(course_len)

        dist_travelled = 0.0
        current_idx = self._calc_nearest_idx(curr_state)

        ref_state[0, 0] = self.c_splines[0][current_idx]
        ref_state[1, 0] = self.c_splines[1][current_idx]
        ref_state[2, 0] = speed_profile[current_idx]
        ref_state[3, 0] = self.c_splines[2][current_idx]

        for t in range(TIME_HORIZON + 1):
            # calculate the distance travelled using velocity state
            dist_travelled += curr_state[2] * DT
            dist_idx = np.round(dist_travelled / self.spline_dist).astype(int)

            if (current_idx + dist_idx) < course_len:
                ref_state[0, t] = self.c_splines[0][current_idx + dist_idx]  # x
                ref_state[1, t] = self.c_splines[1][current_idx + dist_idx]  # y
                ref_state[2, t] = speed_profile[current_idx + dist_idx]  # velocity
                ref_state[3, t] = self.c_splines[2][current_idx + dist_idx]  # yaw
                ref_input[0, t] = 0
            else:
                ref_state[0, t] = self.c_splines[0, course_len - 1]
                ref_state[1, t] = self.c_splines[1, course_len - 1]
                ref_state[2, t] = speed_profile[course_len - 1]
                ref_state[3, t] = self.c_splines[2, course_len - 1]
                ref_input[0, t] = 0

        return ref_state, ref_input

    def _calc_nearest_idx(self, curr_state):
        dx = [curr_state[0] - i_spline for i_spline in self.c_splines[0][self.index:(self.index + N_IDX_SEARCH)]]
        dy = [curr_state[1] - i_spline for i_spline in self.c_splines[1][self.index:(self.index + N_IDX_SEARCH)]]

        dist = [i_dx ** 2 + i_dy ** 2 for i_dx, i_dy in zip(dx, dy)]
        dist_min = min(dist)

        curr_index = dist.index(dist_min)

        return curr_index

    def _model_matrix(self, v, phi, delta):
        """
            Time = < 0.0001
            Calculate the

        """
        a_dt = np.eye(NUM_STATE_VARS, dtype=np.float32)
        a_dt[0, 2] = DT * np.cos(phi)
        a_dt[0, 3] = DT * v * np.sin(phi)
        a_dt[1, 2] = DT * np.sin(phi)
        a_dt[1, 3] = DT * v * np.cos(phi)
        a_dt[3, 2] = DT * np.tan(delta) / CAR_LENGTH

        b_dt = np.zeros((NUM_STATE_VARS, NUM_OUTPUTS))
        b_dt[2, 0] = DT
        b_dt[3, 1] = DT * v / (CAR_LENGTH * np.cos(delta) ** 2)

        c_dt = np.zeros(NUM_STATE_VARS)
        c_dt[0] = DT * v * np.sin(phi) * phi
        c_dt[1] = DT * v * np.cos(phi) * phi
        c_dt[3] = v * delta / (CAR_LENGTH * np.cos(delta) ** 2)

        return a_dt, b_dt, c_dt

    def _compute(self, ref_states, ref_inputs, predict_states, curr_states):
        states = cp.Variable((NUM_STATE_VARS, TIME_HORIZON + 1))
        inputs = cp.Variable((NUM_OUTPUTS, TIME_HORIZON))
        cost = 0.0
        constraints = []

        for t in range(TIME_HORIZON):
            cost += cp.quad_form(inputs[:, t], I_COST)

            if t != 0:
                cost += cp.quad_form(ref_states[:, t] - states[:, t], S_COST)

            A, B, C = self._model_matrix(predict_states[2, t],
                                         predict_states[3, t],
                                         ref_inputs[0, t])

            constraints += [states[:, t + 1] == A * states[:, t] + B * inputs[:, t] + C]

            if t < (TIME_HORIZON - 1):
                cost += cp.quad_form(inputs[:, t + 1] - inputs[:, t], I_COST_DIFF)
                constraints += [cp.abs(inputs[1, t + 1] - inputs[1, t]) <= MAX_DSTEER * DT]

        cost += cp.quad_form(ref_states[:, TIME_HORIZON] - states[:, TIME_HORIZON], S_FINAL)

        constraints += [states[:, 0] == curr_states] # first state is curr state
        constraints += [states[2, :] >= 0] # velocity >= 0
        constraints += [cp.abs(inputs[0, :]) <= 1] # max acc
        constraints += [cp.abs(inputs[1, :]) <= 2] # max steer
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.ECOS, verbose=self.verbose, feastol=1e-2)

        if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
            o_states = np.array((
                states.value[0, :],
                states.value[1, :],
                states.value[2, :],
                states.value[3, :]))

            o_inputs = np.array((
                inputs.value[0, :],
                inputs.value[1, :]))

        else:
            o_states = np.zeros((4, TIME_HORIZON))
            o_inputs = np.zeros((2, TIME_HORIZON))

        self.o_inputs = o_inputs
        self.o_states = o_states

        return o_inputs, o_states

    def _predict(self, curr_state):
        predicted_states = np.zeros([NUM_STATE_VARS, TIME_HORIZON + 1])
        inputs = np.zeros((NUM_OUTPUTS, TIME_HORIZON + 1))

        # Set the t = 0 for predicted_states, inputs to be
        for i in range(NUM_STATE_VARS):
            predicted_states[i, 0] = curr_state[i]

        for i in range(NUM_OUTPUTS):
            inputs[i, 0] = self.o_inputs[i, 0]

        for acc_i, sa_i, i in zip(inputs[0][:], inputs[1][:], range(1, TIME_HORIZON + 1)):
            state = self._update(curr_state, acc_i, sa_i)
            predicted_states[0, i] = state[0]
            predicted_states[1, i] = state[1]
            predicted_states[2, i] = state[2]
            predicted_states[3, i] = state[3]

        return predicted_states

    def _update(self, state, acc, sa):
        state[0] = state[0] + state[3] * np.cos(state[2]) * DT
        state[1] = state[1] + state[3] * np.sin(state[2]) * DT
        state[2] = state[2] + state[3] / CAR_LENGTH * np.tan(sa) * DT
        state[3] = state[3] + acc * DT

        return state

















