# Vanilla RL Benchmarking
## Goals
The goal for this project is to establish a solid benchmark for pure learning algorithms, allowing current and future autonomous MUR engineers to make efficient decisions about the direction of their project. As this is the first yearof the MUR autonomous project, and due to budget and social restrictions, integration of learning algorithms will be limited to MPC-RL models developed by the control team, should these algorithms prove to be optimal. The purpose of having pure learning algorithms tested is to establish a relative performance benchmark.

## Methodology
For benchmarking purposes, a standardized testing and data collection procedure is required. State, observation and reward functions can be changed using the appropriate files in the repository, however changes should be limited to be realistic for an actual car (e.g. state shouldn't be the pixel representation of the render). For every training run, it is important to save all model parameters (i.e. learning rate, epsillon, gamma etc.) and save the NN model weights. Additionally, states, rewards and car information should be saved - enumerated by episode number. The goal is to see how well the car is able to drive around the track.

To summarise, the following items need to be recorded for each timestep:
- State (state information given to the agent)
- Reward (reward information given to the agent)
- x-pos of car (if not included in the state)
- y-pos of car (if not included in the state)
- car linear velocity (if not included in the state)
- steering angle (if not included in the state)

At each episode (enumerated), the following should be saved:
- Timestep information (see above)
- Total reward (score)

For each algorithm, the following should be saved:
- Model hyperparameters
- PyTorch agent (save the NN weights)
- Reward function
- State's provided to the agent (e.g. absolute x-pos, y-pos, velocity)
- Model files (might be best to branch off vanilla-rl for each model)
- Any additional information

## Expectations and Deliverables
Over the following month, the following algorithms should be tested, implemented and optimized to the best of your knowledge:
- DQN
- REINFORCE
- Actor-Critic (any variation - you can use DQN if you can't decide)
- DDPG

Additionally, please save any notes you have about the process. If you have any questions, or would like any help feel free to reach out to Dennis or Joseph! 

