"""
Author: Dennis Wirya
"""

from simulation.environment import Environment
from path_planners.cubic_spline import CubicSplinePlanner
from path_followers.ltv_mpc import LTVMPCFollower

EPISODES = 1
PRINT_STATE_EVERY = 100
RENDER_EPISODE_EVERY = 500


def main():
    time = 0.0
    dt = 0.2
    env = Environment()
    state = env.reset()
    done = False

    N = 10
    Ts = dt

    planner = CubicSplinePlanner(N, Ts)
    follower = LTVMPCFollower(N, Ts)
    action = [0, 0]

    while not done:
        reference = planner.plan(state)
        oa, od = follower.move(state, reference)
        action[0] = oa[0]
        action[1] = od[0]
        state, step_reward, done, _ = env.step(action)

        env.render()

    env.close()

    # env = Environment()

    #
    # # Both Path Planner and Follower have access of the environment, just to
    # # make it easier to access internal simulation states when needed
    # path_planner = IdealPathPlanner(env=env)
    # path_follower = BasicPathFollower()
    #
    # # Need to read the docs on environment.py to understand what states to plot
    # # Additionally, can also plot animated graphs after each time-step in
    # # plan_path() and follow_path() methods
    #
    # for episode in range(EPISODES):
    #     # Initialise state and done
    #     steps = 0   # Track the number of steps when running an episode
    #     state = env.reset()
    #     done = False
    #     should_print = False
    #     should_render = False
    #
    #     if steps % PRINT_STATE_EVERY == 0:
    #         should_print = True
    #
    #     if episode % RENDER_EPISODE_EVERY == 0:
    #         should_render = True
    #
    #     optimal_path = path_planner.plan(state)
    #
    #     while not done:
    #         action = path_follower.follow_path(optimal_path, state)
    #
    #         state, reward, done, _ = env.step(action)
    #
    #         if should_print is True:
    #             print(f"Step {steps}")
    #             print(f"-reward: {reward}")
    #             print(f"-done: {done}")
    #
    #         if should_render is True:
    #             print(f"Episode {episode}")
    #             env.render(mode='human')

    # env.close()
