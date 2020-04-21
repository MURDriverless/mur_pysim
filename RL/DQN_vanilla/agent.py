import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from .dqn import DeepQNetwork


class Agent:
    """
        Vanilla implementation of DQN.
            - No replay buffer
            - Heavily discretized actions (continuous action space ignored)
    """
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, output_dims,
                 steer_res=2, acc_res=2, br_res=2, max_mem_size=250000, epsilon_end=0.05, epsilon_decay=2e-4):
        """

        :param gamma:
        :param epsilon:
        :param lr:
        :param input_dims:
        :param batch_size:
        :param output_dims:
        :param max_mem_size:
        :param epsilon_end:
        :param epsilon_decay:
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.lr = lr

        self.action_space_steer = [i for i in range(steer_res)]
        self.action_space_accelerate = [i for i in range(acc_res)]
        self.action_space_brake = [i for i in range(br_res)]
        self.action_space_len = sum([len(self.action_space_steer),
                                     len(self.action_space_accelerate),
                                     len(self.action_space_brake)])

        self.mem_size = max_mem_size
        self.batch_size = batch_size

        self.mem_counter = 0
        self.iter_counter = 0

        self.Q_eval = DeepQNetwork(lr, input_dims=input_dims, l1_dims=512, l2_dims=256,
                                   output_dims=self.action_space_len)

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory_steer = np.zeros(self.mem_size, dtype=np.int32)
        self.action_memory_acc = np.zeros(self.mem_size, dtype=np.int32)
        self.action_memory_brake = np.zeros(self.mem_size, dtype=np.int32)
        self.terminal_memory = np.zeros((self.mem_size, *input_dims), dtype=np.bool)

    def store_transition(self, state, action, reward, n_state, terminal):
        """

        :param state:
        :param action:
        :param reward:
        :param n_state:
        :param terminal:
        :return:
        """
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.action_memory_steer[index] = action[0]
        self.action_memory_acc[index] = action[1]
        self.action_memory_brake[index] = action[2]
        self.reward_memory[index] = reward
        self.new_state_memory[index] = n_state
        self.terminal_memory[index] = terminal

    def choose_action(self, observation):





