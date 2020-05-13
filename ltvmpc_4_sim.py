import math
import numpy as np
import cvxpy
import matplotlib.pyplot as plt
import path_planners.cubic_spline as cubic_spline_planner

NX = 4  # x, y, v, yaw
NU = 2  # a, delta
L = 2.5  # [m]

MAX_STEER = np.deg2rad(45.0)  # maximum steering angle [rad]
MAX_DSTEER = np.deg2rad(30.0)  # maximum steering speed [rad/s]
MAX_SPEED = 55.0 / 3.6  # maximum speed [m/s]
MIN_SPEED = -20.0 / 3.6  # minimum speed [m/s]
MAX_ACCEL = 1.0  # maximum accel [m/ss]
MIN_ACCEL = -1.0  # maximum braking [m/ss]

# ----------------------------------------------------------
# LTV MPC stuffs                                           |
# ----------------------------------------------------------


def kinematic_bicycle_c(x, u):
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


def kinematic_bicycle_d(xk, uk, Ts, repeat_sampling=False):
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
            dxdt = kinematic_bicycle_c(xk1, uk)
            xk1 = np.add(xk1, dt * dxdt)
    else:
        dt = Ts
        dxdt = kinematic_bicycle_c(xk, uk)
        xk1 = np.add(xk, dt * dxdt)
    return xk1


def linearise_kinematic_bicycle_d(v, yaw, delta, Ts):
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


def calculate_xbar(x0, oa, od, Ts, horizon_length):
    """
    Computes xbar for the range [0, horizon_length] (inclusive), by applying
    optimal acceleration and delta at each iteration

    Args:
        x0 (numpy.ndarray): 1D vector of size NX containing current state
        oa (numpy.ndarray): 1D vector of size horizon_length listing optimal acceleration
        od (numpy.ndarray): 1D vector of size horizon_length listing optimal delta
        Ts (float): sampling period
        horizon_length (int): prediction horizon length

    Returns:
        numpy.ndarray: 2D vector of size (NX, horizon_length + 1)
    """
    xbar = np.zeros((NX, horizon_length + 1))
    xk = np.zeros(NX)
    uk = np.zeros(NU)
    uk[0] = oa[0]
    uk[1] = od[0]

    # At horizon = 0, xbar is the current state, which is x0
    for i in range(len(x0)):
        xbar[i, 0] = x0[i]
        xk[i] = x0[i]

    # In the next horizons, calculate xbar by applying optimal oa and od to the plant
    # Note that we set end range to be "horizon_length + 1" as we are iterating from 1.
    # This is because the first horizon is the current measurements
    for (ai, di, i) in zip(oa, od, range(1, horizon_length + 1)):
        # Apply input to plant
        x, y, v, yaw = kinematic_bicycle_d(xk, uk, Ts)
        # Update state from plant
        xk[0] = x
        xk[1] = y
        xk[2] = v
        xk[3] = yaw
        # Update optimal inputs for next iteration
        uk[0] = ai
        uk[1] = di
        # Set xbar for subsequent horizons to be exactly the states from applying
        # optimal inputs
        xbar[0, i] = xk[0]
        xbar[1, i] = xk[1]
        xbar[2, i] = xk[2]
        xbar[3, i] = xk[3]
    return xbar


def iterative_ltv_mpc(x0, xref, dref, oa, od, Ts, horizon_length):
    MAX_ITER = 3
    DU_TH = 0.1  # iteration finish param
    ox, oy, ov, oyaw = 0.0, 0.0, 0.0, 0.0
    for i in range(MAX_ITER):
        xbar = calculate_xbar(x0, oa, od, Ts, horizon_length)
        prev_oa, prev_od = oa[:], od[:]
        ox, oy, ov, oyaw, oa, od = ltv_mpc_control(x0, xref, dref, xbar, Ts, horizon_length)
        du = sum(abs(oa - prev_oa)) + sum(abs(od - prev_od))  # calc u change value
        if du <= DU_TH:
            break
    else:
        print("Iterative is max iter")

    return ox, oy, ov, oyaw, oa, od


def ltv_mpc_control(x0, xref, dref, xbar, Ts, horizon_length):
    """
    Computes optimal acceleration and delta for subsequent horizons, given the current state, states reference,
    steering input reference and the equilibrium value xbar to linearise the car kinematics

    Args:
        x0 (numpy.ndarray): 1D vector of size NX containing the current state
        xref (numpy.ndarray): 2D vector of size (NX, horizon_length + 1) containing the current state reference
        dref (numpy.ndarray): 2D vector of size (1, horizon_length) containing the current steering reference
        xbar (numpy.ndarray): 2D vector of size (NX, horizon_length+1) containing the equilibrium values at each horizon
        Ts (float): sampling period (or dt)
        horizon_length (int): prediction horizon length

    Returns:
        np.ndarray: 2D vector of size (NX, horizon_length) -> [oa; ow]
            It lists the optimal acceleration and optimal delta for horizons up to horizon_length
    """
    Q = np.diag([1.0, 1.0, 0.5, 0.5])  # state cost matrix
    Qf = Q  # final state cost matrix
    R = np.diag([0.01, 0.01])  # input cost matrix
    Rd = np.diag([0.01, 1.0])  # input difference cost matrix

    x = cvxpy.Variable((NX, horizon_length + 1))
    u = cvxpy.Variable((NU, horizon_length))
    cost = 0.0
    constraints = []

    for t in range(horizon_length):
        cost += cvxpy.quad_form(u[:, t], R)

        # Penalise state deviation from reference
        if t != 0:
            cost += cvxpy.quad_form(xref[:, t] - x[:, t], Q)

        A, B, C = linearise_kinematic_bicycle_d(xbar[2, t], xbar[3, t], dref[0, t], Ts)
        constraints += [x[:, t+1] == A * x[:, t] + B * u[:, t] + C]

        if t < (horizon_length - 1):
            cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], Rd)
            # Saturate the changes in steering from previous state
            constraints += [cvxpy.abs(u[1, t + 1] - u[1, t]) <= MAX_DSTEER * Ts]

    # At the end of the horizon, add cost of Qf
    cost += cvxpy.quad_form(xref[:, horizon_length] - x[:, horizon_length], Qf)

    # Initial state must be the same as x0
    constraints += [x[:, 0] == x0]
    # Constrain speed
    constraints += [x[2, :] <= MAX_SPEED]
    constraints += [x[2, :] >= MIN_SPEED]
    # Constrain acceleration
    constraints += [u[0, :] <= MAX_ACCEL]
    constraints += [u[0, :] >= MIN_ACCEL]
    # Constrain delta
    constraints += [cvxpy.abs(u[1, :]) <= MAX_STEER]

    # Solve optimisation problem
    problem = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    problem.solve(solver=cvxpy.ECOS, verbose=False)

    if problem.status == cvxpy.OPTIMAL or problem.status == cvxpy.OPTIMAL_INACCURATE:
        ox = np.array(x.value[0, :]).flatten()
        oy = np.array(x.value[1, :]).flatten()
        ov = np.array(x.value[2, :]).flatten()
        oyaw = np.array(x.value[3, :]).flatten()
        oa = np.array(u.value[0, :]).flatten()
        od = np.array(u.value[1, :]).flatten()
    else:
        print("No solution")
        ox, oy, ov, oyaw, oa, od = None, None, None, None, None, None
    return ox, oy, ov, oyaw, oa, od


# ----------------------------------------------------------
# Simulation stuffs                                        |
# ----------------------------------------------------------


def get_switch_back_course(dl):
    ax = [0.0, 30.0, 6.0, 20.0, 35.0]
    ay = [0.0, 0.0, 20.0, 35.0, 20.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)
    ax = [35.0, 10.0, 0.0, 0.0]
    ay = [20.0, 30.0, 5.0, 0.0]
    cx2, cy2, cyaw2, ck2, s2 = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)
    cyaw2 = [i - math.pi for i in cyaw2]
    cx.extend(cx2)
    cy.extend(cy2)
    cyaw.extend(cyaw2)
    ck.extend(ck2)
    return cx, cy, cyaw, ck


def pi_2_pi(angle):
    while angle > math.pi:
        angle = angle - 2.0 * math.pi
    while angle < -math.pi:
        angle = angle + 2.0 * math.pi
    return angle


def calc_nearest_index(state, cx, cy, cyaw, pind):
    N_IND_SEARCH = 10  # Search index number
    x = state[0]
    y = state[1]
    dx = [x - icx for icx in cx[pind:(pind + N_IND_SEARCH)]]
    dy = [y - icy for icy in cy[pind:(pind + N_IND_SEARCH)]]
    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]
    mind = min(d)
    ind = d.index(mind) + pind
    mind = math.sqrt(mind)
    dxl = cx[ind] - x
    dyl = cy[ind] - y
    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        mind *= -1
    return ind, mind


def calc_ref_trajectory(x0, cx, cy, cyaw, sp, dl, pind, Ts, horizon_length):
    """
    Generates states and steering input reference based on the centre positions of the track and speed profile,
    up to horizon length

    Args:
        x0 (numpy.ndarray): 1D array of length NX, which is the current state measurements
        cx (numpy.ndarray): course x - 1D array of unknown length
        cy (numpy.ndarray): course y - 1D array of unknown length, same size as cx
        cyaw (numpy.ndarray): course yaw - 1D array of unknown length
        sp (numpy.ndarray): speed profile - 1D array of unknown length, presumably the same size as cx
        dl (float): course tick
        pind (int): position index
        Ts (float): controller sampling period
        horizon_length (int): prediction horizon length

    Returns:
        numpy.ndarray: [xref, dref, tind]
            - xref (numpy.ndarray): state reference, 2D matrix of size (NX, horizon_length + 1)
            - dref (numpy.ndarray): steering input reference, 1D array of of length (horizon_length + 1)
            - tind (int): nearest next forward (front of car) index in the course where the car is currently at
    """
    xref = np.zeros((NX, horizon_length + 1))
    dref = np.zeros((1, horizon_length + 1))
    ncourse = len(cx)
    ind, _ = calc_nearest_index(x0, cx, cy, cyaw, pind)

    if pind >= ind:
        ind = pind

    xref[0, 0] = cx[ind]
    xref[1, 0] = cy[ind]
    xref[2, 0] = sp[ind]
    xref[3, 0] = cyaw[ind]
    dref[0, 0] = 0.0  # steer operational point should be 0
    travel = 0.0

    for i in range(horizon_length + 1):
        travel += abs(x0[2]) * Ts
        dind = int(round(travel / dl))
        if (ind + dind) < ncourse:
            xref[0, i] = cx[ind + dind]
            xref[1, i] = cy[ind + dind]
            xref[2, i] = sp[ind + dind]
            xref[3, i] = cyaw[ind + dind]
            dref[0, i] = 0.0
        else:
            xref[0, i] = cx[ncourse - 1]
            xref[1, i] = cy[ncourse - 1]
            xref[2, i] = sp[ncourse - 1]
            xref[3, i] = cyaw[ncourse - 1]
            dref[0, i] = 0.0
    return xref, dref, ind


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


def check_goal(state, goal, tind, nind):
    GOAL_DIS = 1.5  # goal distance
    STOP_SPEED = 0.5 / 3.6  # stop speed

    x = state[0]
    y = state[1]
    v = state[2]

    # check goal
    dx = x - goal[0]
    dy = y - goal[1]
    d = math.hypot(dx, dy)

    is_goal = (d <= GOAL_DIS)

    if abs(tind - nind) >= 5:
        is_goal = False

    is_stop = (abs(v) <= STOP_SPEED)

    if is_goal and is_stop:
        return True
    else:
        return False


def plot_car(x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):  # pragma: no cover
    # Vehicle parameters
    LENGTH = 4.5  # [m]
    WIDTH = 2.0  # [m]
    BACKTOWHEEL = 1.0  # [m]
    WHEEL_LEN = 0.3  # [m]
    WHEEL_WIDTH = 0.2  # [m]
    TREAD = 0.7  # [m]
    WB = L  # [m]

    outline = np.array([[-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
                        [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                         [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD]])

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


def run_simulation(cx, cy, cyaw, sp, dl, x0, Ts, horizon_length, show_animation):
    MAX_TIME = 500  # max simulation time
    goal = [cx[-1], cy[-1]]

    # initial yaw compensation
    if x0[3] - cyaw[0] >= math.pi:
        x0[3] -= math.pi * 2.0
    elif x0[3] - cyaw[0] <= -math.pi:
        x0[3] += math.pi * 2.0

    time = 0.0
    t = [0.0]
    x = [x0[0]]
    y = [x0[1]]
    v = [x0[2]]
    yaw = [x0[3]]
    acceleration = [0.0]
    delta = [0.0]
    target_ind, _ = calc_nearest_index(x0, cx, cy, cyaw, 0)
    oa = np.zeros(horizon_length)  # optimal acceleration
    od = np.zeros(horizon_length)  # optimal delta
    xk = x0
    uk = np.zeros(2)
    uk[0] = 0.0
    uk[1] = 0.0

    cyaw = smooth_yaw(cyaw)

    while time <= MAX_TIME:
        xref, dref, target_ind = calc_ref_trajectory(xk, cx, cy, cyaw, sp, dl, target_ind, Ts, horizon_length)

        ox, oy, ov, oyaw, oa, od = iterative_ltv_mpc(xk, xref, dref, oa, od, Ts, horizon_length)
        # xbar = calculate_xbar(xk, oa, od, Ts, horizon_length)
        # ox, oy, ov, oyaw, oa, od = ltv_mpc_control(xk, xref, dref, xbar, Ts, horizon_length)

        if oa is not None:
            ai, di = oa[0], od[0]
        else:
            ai, di = 0.0, 0.0

        print(f"vref={xref[0][2]}, v={xk[2]}\n")

        # Update plant
        uk[0] = ai
        uk[1] = di
        xk = kinematic_bicycle_d(xk, uk, Ts)
        time += Ts

        t.append(time)
        x.append(xk[0])
        y.append(xk[1])
        v.append(xk[2])
        yaw.append(xk[3])
        acceleration.append(ai)
        delta.append(di)

        if check_goal(xk, goal, target_ind, len(cx)):
            print("Goal")
            break

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            if ox is not None:
                plt.plot(ox, oy, "xr", label="MPC")
            plt.plot(cx, cy, "-r", label="course")
            plt.plot(x, y, "ob", label="trajectory")
            plt.plot(xref[0, :], xref[1, :], "xk", label="xref")
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            plot_car(xk[0], xk[1], xk[3], steer=di)
            plt.axis("equal")
            plt.grid(True)
            plt.title("Time[s]:" + str(round(time, 2))
                      + ", speed[km/h]:" + str(round(xk[2] * 3.6, 2)))
            plt.pause(0.0001)

    return t, x, y, yaw, v, delta, acceleration


def main():
    print(__file__ + " start!!")

    dl = 1.0  # course tick
    TARGET_SPEED = 10.0 / 3.6  # [m/s] target speed

    cx, cy, cyaw, ck = get_switch_back_course(dl)
    sp = calc_speed_profile(cx, cy, cyaw, TARGET_SPEED)

    x0 = np.zeros(NX)
    x0[0] = cx[0]
    x0[1] = cy[0]
    x0[2] = 0.0
    x0[3] = 0.0

    Ts = 0.2
    horizon_length = 5
    show_animation = True

    t, x, y, yaw, v, d, a = run_simulation(cx, cy, cyaw, sp, dl, x0, Ts, horizon_length, show_animation)

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
