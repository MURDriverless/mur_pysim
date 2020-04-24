import car_racing
from utils import track, checkpoints
from RL.DQN_vanilla.agent import Agent

import torch
import numpy as np

PATH = 'saves/dqn_model2.txt'

if __name__ == "__main__":
    env = car_racing.CarRacing(load_track=True)
    agent = Agent(lr=0.01, gamma=0.08, epsilon=0.2, input_dims=[5], batch_size=512)
    env.reset()
    track_xy = track.Coordinates.load()
    cp = checkpoints.Checkpoint(track_xy)
    num_games = 10000
    scores = []

    for i in range(num_games):
        score = 0
        done = False
        reward = 0
        cp.index = 0
        cp_index = cp.index
        cp_last_index = cp_index
        steps_since_cp = 0
        observation = np.zeros(5, dtype=np.float32)

        while not done:
            env.render()
            # Choose action
            action = agent.choose_action(observation)
            a = [0, action[1], 0]

            if action[0] < 1:
                a[0] = -1
            elif action[0] > 1:
                a[0] = 1
            else:
                action[0] = 0

            # Step environment
            _, r, _, _ = env.step(a)

            # Calculate observation_, reward, done
            # observation_
            car_x = env.car.hull.position[0]
            car_y = env.car.hull.position[1]
            ob_dist = cp.check_dist(car_x, car_y)
            ob_x, ob_y, ob_theta = cp.check_xytheta_dist(car_x, car_y)
            vel_x = env.car.hull.linearVelocity[0]
            vel_y = env.car.hull.linearVelocity[1]
            vel = np.sqrt(vel_x ** 2 + vel_y ** 2)

            observation_ = np.array([ob_x, ob_y, ob_dist, ob_theta, vel], dtype=np.float32)

            # reward
            if ob_dist < 5:
                reward += (5 - ob_dist) * 0.1 + vel * 0.1

            if cp_index == cp_last_index:
                steps_since_cp += 1
            else:
                reward += 1000

            # done
            if steps_since_cp > 1000:
                done = True

            score += reward

            # agent.store_transition()
            agent.store_transition(observation, action, reward,
                                   observation_, done)

            # agent.learn()
            agent.learn()

            observation = observation_
            cp_last_index = cp_index

        env.reset()
        scores.append((i, score, cp.index))
        print(f"Episode: {i + 1}\n",
              f"Score: {score}\n",
              f"Avg Score: {np.mean(scores[-100:])}\n",
              f"Epsilon: {agent.epsilon}\n",
              f"Checkpoint Reached: {cp.index}")

        if i % 100 == 0:
            with open(PATH, 'w') as file:
                file.write(str(scores))



