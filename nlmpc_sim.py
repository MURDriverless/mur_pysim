import math
import numpy as np
import matplotlib.pyplot as plt
import cvxpy
import path_planners.cubic_spline as cubic_spline_planner

NX = 5  # x = x, y, v, yaw
NU = 2  # a = [accel, steer]
HORIZON = 15  # horizon length

GOAL_DIS = 1.5  # goal distance
STOP_SPEED = 0.5 / 3.6  # stop speed
MAX_TIME = 500.0  # max simulation time
DT = 0.2  # [s] time tick
show_animation = True

# Vehicle parameters
LENGTH = 4.5  # [m]
WIDTH = 2.0  # [m]
BACKTOWHEEL = 1.0  # [m]
WHEEL_LEN = 0.3  # [m]
WHEEL_WIDTH = 0.2  # [m]
TREAD = 0.7  # [m]
WB = 2.5  # [m]

MIN_STEER = 0.0
MAX_STEER = np.deg2rad(45.0)  # maximum steering angle [rad]
MAX_DSTEER = np.deg2rad(30.0)  # maximum steering speed [rad/s]
MAX_D2STEER = np.deg2rad(2.0)  # maximum change in steering speed (input) [rad/s]
MAX_SPEED = 55.0 / 3.6  # maximum speed [m/s]
MIN_SPEED = -20.0 / 3.6  # minimum speed [m/s]
MAX_ACCEL = 1.0  # maximum accel [m/ss]
MAX_BRAKE = -1.0

TARGET_SPEED = 10.0 / 3.6  # [m/s] target speed
N_IND_SEARCH = 10  # Search index number


class State:
    def __init__(self, x=0.0, y=0.0, v=0.0, yaw=0.0, delta=0.0):
        self.x = x
        self.y = y
        self.v = v
        self.yaw = yaw
        self.delta = delta

    def to_list(self):
        return [self.x, self.y, self.v, self.yaw, self.delta]


def pi_2_pi(angle):
    while angle > math.pi:
        angle = angle - 2.0 * math.pi
    while angle < -math.pi:
        angle = angle + 2.0 * math.pi
    return angle


def plot_car(x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):
    outline = np.array([[-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
                        [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                         [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD,
                          -WHEEL_WIDTH - TREAD]])

    rr_wheel = np.copy(fr_wheel)

    fl_wheel = np.copy(fr_wheel)
    fl_wheel[1, :] *= -1
    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1

    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                     [-math.sin(yaw), math.cos(yaw)]])
    Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                     [-math.sin(steer), math.cos(steer)]])

    fr_wheel = (fr_wheel.T.dot(Rot2)).T
    fl_wheel = (fl_wheel.T.dot(Rot2)).T
    fr_wheel[0, :] += WB
    fl_wheel[0, :] += WB

    fr_wheel = (fr_wheel.T.dot(Rot1)).T
    fl_wheel = (fl_wheel.T.dot(Rot1)).T

    outline = (outline.T.dot(Rot1)).T
    rr_wheel = (rr_wheel.T.dot(Rot1)).T
    rl_wheel = (rl_wheel.T.dot(Rot1)).T

    outline[0, :] += x
    outline[1, :] += y
    fr_wheel[0, :] += x
    fr_wheel[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    fl_wheel[0, :] += x
    fl_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y

    plt.plot(np.array(outline[0, :]).flatten(),
             np.array(outline[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fr_wheel[0, :]).flatten(),
             np.array(fr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rr_wheel[0, :]).flatten(),
             np.array(rr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fl_wheel[0, :]).flatten(),
             np.array(fl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rl_wheel[0, :]).flatten(),
             np.array(rl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(x, y, "*")


def kinematic_bicycle_c(x, u):
    dxdt = np.zeros(5)
    dxdt[0] = x[2] * np.cos(x[3])  # xdot
    dxdt[1] = x[2] * np.sin(x[3])  # ydot
    dxdt[2] = 0.5 * u[0]  # throttle
    dxdt[3] = (x[2] * np.tan(x[4])) / BACKTOWHEEL  # yaw
    dxdt[4] = u[1]  # steering
    return dxdt


def kinematic_bicycle_d(xk, uk, dt):
    # Constrain inputs
    # if uk[1] >= MAX_STEER:
    #     uk[1] = MAX_STEER
    # elif uk[1] <= MIN_STEER:
    #     uk[1] = MIN_STEER
    dxdt = kinematic_bicycle_c(xk, uk)
    return np.add(xk, dt * dxdt)


def plant_dynamics_d(xk, uk, dt):
    """
    Returns the

    Args:
        xk:
        uk:
        dt:

    Returns:

    """


def calc_speed_profile(cx, cy, cyaw, target_speed):
    speed_profile = [target_speed] * len(cx)
    direction = 1.0  # forward
    # Set stop point
    for i in range(len(cx) - 1):
        dx = cx[i + 1] - cx[i]
        dy = cy[i + 1] - cy[i]
        move_direction = math.atan2(dy, dx)
        if dx != 0.0 and dy != 0.0:
            dangle = abs(pi_2_pi(move_direction - cyaw[i]))
            if dangle >= math.pi / 4.0:
                direction = -1.0
            else:
                direction = 1.0
        if direction != 1.0:
            speed_profile[i] = - target_speed
        else:
            speed_profile[i] = target_speed
    speed_profile[-1] = 0.0
    return speed_profile


def get_straight_course3(dl):
    ax = [0.0, -10.0, -20.0, -40.0, -50.0, -60.0, -70.0]
    ay = [0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)
    cyaw = [i - math.pi for i in cyaw]
    return cx, cy, cyaw, ck


def calc_nearest_index(state, cx, cy, cyaw, pind):
    dx = [state.x - icx for icx in cx[pind:(pind + N_IND_SEARCH)]]
    dy = [state.y - icy for icy in cy[pind:(pind + N_IND_SEARCH)]]
    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]
    mind = min(d)
    ind = d.index(mind) + pind
    mind = math.sqrt(mind)
    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y
    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        mind *= -1
    return ind, mind


def smooth_yaw(yaw):
    for i in range(len(yaw) - 1):
        dyaw = yaw[i + 1] - yaw[i]
        while dyaw >= math.pi / 2.0:
            yaw[i + 1] -= math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]
        while dyaw <= -math.pi / 2.0:
            yaw[i + 1] += math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]
    return yaw


def calc_ref_trajectory(state, cx, cy, cyaw, ck, sp, dl, pind):
    xref = np.zeros((NX, HORIZON + 1))
    # dref = np.zeros((1, T + 1))
    ncourse = len(cx)

    ind, _ = calc_nearest_index(state, cx, cy, cyaw, pind)

    if pind >= ind:
        ind = pind

    xref[0, 0] = cx[ind]
    xref[1, 0] = cy[ind]
    xref[2, 0] = sp[ind]
    xref[3, 0] = cyaw[ind]
    # dref[0, 0] = 0.0  # steer operational point should be 0
    xref[4, 0] = 0.0  # steer operational point should be 0

    travel = 0.0

    for i in range(HORIZON + 1):
        travel += abs(state.v) * DT
        dind = int(round(travel / dl))

        if (ind + dind) < ncourse:
            xref[0, i] = cx[ind + dind]
            xref[1, i] = cy[ind + dind]
            xref[2, i] = sp[ind + dind]
            xref[3, i] = cyaw[ind + dind]
            xref[4, i] = 0.0
        else:
            xref[0, i] = cx[ncourse - 1]
            xref[1, i] = cy[ncourse - 1]
            xref[2, i] = sp[ncourse - 1]
            xref[3, i] = cyaw[ncourse - 1]
            xref[4, i] = 0.0

    return xref, ind


def nonlinear_mpc_control(x0, xref):
    """
    Computes the optimal state and actuation inputs

    Returns:
        numpy.ndarray: Contains the tuple elements
            (optimal_x, optimal_y, optimal_v, optimal_yaw, optimal_throttle, optimal_steering)
    """

    # Set x to be a matrix of size row (number of states) and column (how far we want to predict)
    x = cvxpy.Variable((NX, HORIZON))
    # Set u to be a matrix of size row (number of inputs) and column (how far we want to predict)
    u = cvxpy.Variable((NU, HORIZON))

    cost = 0.0
    constraints = []

    # Tracking error weights
    Q = np.diag([1.0, 1.0, 0.5, 0.5])
    # Final state tracking error weights
    Qf = np.diag([1.0, 1.0, 0.5, 0.5])
    # Input value weights
    R = np.diag([0.1, 0.1])
    # Input difference weights
    Rd = np.diag([0.1, 1])

    # We want cost to be
    for t in range(HORIZON):
        tracking_error = xref[:, t] - x[:, t]
        input_change = u[:, t+1] - u[:, t]

        # Penalise high inputs
        cost += cvxpy.quad_form(u[:, t], R)

        # Penalise deviation from reference state
        if t != 0:
            cost += cvxpy.quad_form(tracking_error, Q)

        # Constrain the next states to follow the predictive model
        xk1 = kinematic_bicycle_d(x[:, t], u[:, t], DT)
        constraints += [x[:, t + 1] == xk1]

        # Penalise large input difference
        if t < HORIZON - 1:
            cost += cvxpy.quad_form(input_change, Rd)
            # Constrain max throttle and max steering
            constraints += [cvxpy.abs(input_change[1, t]) <= MAX_DSTEER * DT]

    cost += cvxpy.quad_form(xref[:, HORIZON] - x[:, HORIZON], Qf)

    constraints += [x[:, 0] == x0]
    constraints += [x[2, :] <= MAX_SPEED]
    constraints += [x[2, :] >= MIN_SPEED]
    constraints += [cvxpy.abs(x[3, :]) <= MAX_STEER]
    constraints += [u[0, :] <= MAX_ACCEL]
    constraints += [u[0, :] >= MAX_BRAKE]
    constraints += [cvxpy.abs(u[1, :]) <= MAX_DSTEER]

    # Formulate problem
    problem = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    problem.solve(solver=cvxpy.ECOS, verbose=False)

    if problem.status == cvxpy.OPTIMAL or problem.status == cvxpy.OPTIMAL_INACCURATE:
        optimal_x = np.array(x.value[0, :]).flatten()
        optimal_y = np.array(x.value[1, :]).flatten()
        optimal_v = np.array(x.value[2, :]).flatten()
        optimal_yaw = np.array(x.value[3, :]).flatten()
        optimal_delta = np.array(x.value[4, :]).flatten()
        optimal_throttle = np.array(u.value[0, :]).flatten()
        optimal_steering = np.array(u.value[1, :]).flatten()
    else:
        optimal_x, optimal_y, optimal_v, optimal_yaw, optimal_delta, optimal_throttle, optimal_steering =\
            None, None, None, None, None, None, None

    return optimal_x, optimal_y, optimal_v, optimal_yaw, optimal_delta, optimal_throttle, optimal_steering


def check_goal(state, goal, tind, nind):
    # check goal
    dx = state.x - goal[0]
    dy = state.y - goal[1]
    d = math.hypot(dx, dy)
    isgoal = (d <= GOAL_DIS)
    if abs(tind - nind) >= 5:
        isgoal = False
    isstop = (abs(state.v) <= STOP_SPEED)
    if isgoal and isstop:
        return True
    return False


def run_simulation(cx, cy, cyaw, ck, sp, dl, initial_state, show_animation=True):
    """
    Simulation
    cx: course x position list
    cy: course y position list
    cy: course yaw position list
    ck: course curvature list
    sp: speed profile
    dl: course tick [m]
    """
    goal = [cx[-1], cy[-1]]
    state = initial_state

    # initial yaw compensation
    if state.yaw - cyaw[0] >= math.pi:
        state.yaw -= math.pi * 2.0
    elif state.yaw - cyaw[0] <= -math.pi:
        state.yaw += math.pi * 2.0

    time = 0.0
    x = [state.x]
    y = [state.y]
    v = [state.v]
    yaw = [state.yaw]
    delta = [state.delta]
    t = [0.0]
    steering = [0.0]
    throttle = [0.0]
    target_ind, _ = calc_nearest_index(state, cx, cy, cyaw, 0)

    optimal_throttle, optimal_steering = None, None
    cyaw = smooth_yaw(cyaw)

    while MAX_TIME >= time:
        xref, target_ind = calc_ref_trajectory(state, cx, cy, cyaw, ck, sp, dl, target_ind)

        x0 = [state.x, state.y, state.v, state.yaw, state.delta]  # current state

        optimal_x, optimal_y, optimal_v, optimal_yaw, optimal_delta, optimal_throttle, optimal_steering =\
            nonlinear_mpc_control(x0, xref)

        # oa, odelta, ox, oy, oyaw, ov = iterative_linear_mpc_control(
        #     xref, x0, dref, oa, odelta)

        if optimal_steering is not None:
            throttle_i, steer_i = optimal_throttle[0], optimal_steering[0]
        else:
            throttle_i, steer_i = 0.0, 0.0

        state = kinematic_bicycle_d(state.to_list(), throttle_i, steer_i)
        time = time + DT

        x.append(state.x)
        y.append(state.y)
        v.append(state.v)
        yaw.append(state.yaw)
        delta.append(state.delta)
        t.append(time)
        throttle.append(throttle_i)
        steering.append(steer_i)

        if check_goal(state, goal, target_ind, len(cx)):
            print("Goal")
            break

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                                         lambda event: [exit(0) if event.key == 'escape' else None])
            if optimal_x is not None:
                plt.plot(optimal_x, optimal_y, "xr", label="MPC")
            plt.plot(cx, cy, "-r", label="course")
            plt.plot(x, y, "ob", label="trajectory")
            plt.plot(xref[0, :], xref[1, :], "xk", label="xref")
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            plot_car(state.x, state.y, state.yaw, steer=steer_i)
            plt.axis("equal")
            plt.grid(True)
            plt.title("Time[s]:" + str(round(time, 2))
                      + ", speed[km/h]:" + str(round(state.v * 3.6, 2)))
            plt.pause(0.0001)

    return t, x, y, yaw, v, steering, throttle


def main():
    print(__file__ + " start!!")

    dl = 1.0  # course tick
    cx, cy, cyaw, ck = get_straight_course3(dl)
    sp = calc_speed_profile(cx, cy, cyaw, TARGET_SPEED)
    initial_state = State(x=cx[0], y=cy[0], yaw=0.0, v=0.0, delta=0.0)

    t, x, y, yaw, v, d, a = run_simulation(cx, cy, cyaw, ck, sp, dl, initial_state)

    if show_animation:  # pragma: no cover
        plt.close("all")
        plt.subplots()
        plt.plot(cx, cy, "-r", label="spline")
        plt.plot(x, y, "-g", label="tracking")
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.legend()

        plt.subplots()
        plt.plot(t, v, "-r", label="speed")
        plt.grid(True)
        plt.xlabel("Time [s]")
        plt.ylabel("Speed [kmh]")

        plt.show()


if __name__ == "__main__":
    main()
