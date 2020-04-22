import car_racing
from utils import track, checkpoints
from RL.DQN_vanilla.agent import Agent

import numpy as np

if __name__ == "__main__":
    env = car_racing.CarRacing(load_track=True)
    agent = Agent(lr=0.1, gamma=0.1, epsilon=0.1, input_dims=[4], batch_size=256)
    env.reset()
    track_xy = track.Coordinates.load()
    cp = checkpoints.Checkpoint(track_xy)
    num_games = 500
    scores = []

    for i in range(num_games):
        score = 0
        done = False
        reward = 0
        cp_index = cp.index
        cp_last_index = cp_index
        steps_since_cp = 0
        observation = np.zeros(4, dtype=np.float32)
        env.reset()

        while not done:
            env.render()
            # Choose action
            action = agent.choose_action(observation)

            # Step environment
            _, r, _, _ = env.step(action)

            # Calculate observation_, reward, done
            # observation_
            car_x = env.car.hull.position[0]
            car_y = env.car.hull.position[1]
            ob_dist = cp.check_dist(car_x, car_y)
            ob_x, ob_y, ob_theta = cp.check_xytheta_dist(car_x, car_y)
            observation_ = np.array([ob_x, ob_y, ob_dist, ob_theta], dtype=np.float32)

            # reward
            if ob_dist > 5:
                reward -= 0.01
            else:
                reward += ob_dist * 0.001

            if action[1] > 5:
                reward -= 1

            # done
            if cp_index == cp_last_index:
                steps_since_cp += 1

            if steps_since_cp > 250:
                done = True

            score += reward

            # agent.store_transition()
            agent.store_transition(observation, action, reward,
                                   observation_, done)

            # agent.learn()
            agent.learn()

            observation = observation_
            cp_last_index = cp_index

        print(f"Episode: {i + 1}\n",
              f"Score: {score}\n",
              f"Avg Score: {np.mean(scores[-100:])}\n",
              f"Epsilon: {agent.epsilon}")









