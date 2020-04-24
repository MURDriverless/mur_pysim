import torch as T
import numpy as np

from .dqn import DeepQNetwork


class Agent:
    """
        Vanilla implementation of DQN.
            - No replay buffer
            - Heavily discretized actions (continuous action space ignored)
    """
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size,
                 steer_res=3, acc_res=10, br_res=2, max_mem_size=500000, epsilon_min=0.01, epsilon_decay=1e-5):
        """

        :param gamma:
        :param epsilon:
        :param lr:
        :param input_dims:
        :param batch_size:
        :param max_mem_size:
        :param epsilon_end:
        :param epsilon_decay:
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = lr

        self.action_space_steer = [i for i in range(steer_res)]
        self.action_space_accelerate = [i for i in range(1, acc_res)]
        self.action_space_brake = [i for i in range(br_res)]

        self.mem_size = max_mem_size
        self.batch_size = batch_size

        self.mem_counter = 0
        self.iter_counter = 0

        self.Q_eval = DeepQNetwork(lr, input_dims=input_dims, l1_dims=128, l2_dims=256, l3_dims=512,
                                   steer_dims=len(self.action_space_steer),
                                   acc_dims=len(self.action_space_accelerate),
                                   brake_dims=len(self.action_space_brake))

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory_steer = np.zeros(self.mem_size, dtype=np.int32)
        self.action_memory_acc = np.zeros(self.mem_size, dtype=np.int32)
        self.action_memory_brake = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.bool)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

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

        self.mem_counter += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.Q_eval.device)
            actions_steer, actions_acc, actions_brake = self.Q_eval.forward(state)
            action_steer = T.argmax(actions_steer)
            action_acc = T.argmax(actions_acc)
            action_brake = T.argmax(actions_brake)

        else:
            action_steer = np.random.choice(self.action_space_steer)
            action_acc = np.random.choice(self.action_space_accelerate[1:])
            action_brake = np.random.choice(self.action_space_brake)

        action = np.array([action_steer, action_acc, action_brake], dtype=np.int)
        return action

    def learn(self):
        if self.mem_counter < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_counter, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        action_batch_steer = self.action_memory_steer[batch]
        action_batch_acc = self.action_memory_acc[batch]
        action_batch_brake = self.action_memory_brake[batch]
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        q_eval_steer, q_eval_acc, q_eval_brake = self.Q_eval.forward(state_batch)

        q_eval_steer = q_eval_steer[batch_index, action_batch_steer]
        q_eval_acc = q_eval_acc[batch_index, action_batch_acc]
        q_eval_brake = q_eval_brake[batch_index, action_batch_brake]

        q_next_steer, q_next_acc, q_next_brake = self.Q_eval.forward(new_state_batch)
        q_next_steer[terminal_batch], q_next_acc[terminal_batch], q_next_acc[terminal_batch] = 0.0, 0.0, 0.0

        q_target_steer = reward_batch + self.gamma * T.max(q_next_steer, dim=1)[0]
        q_target_acc = reward_batch + self.gamma * T.max(q_next_acc, dim=1)[0]
        q_target_brake = reward_batch + self.gamma * T.max(q_next_brake, dim=1)[0]

        loss_steer = self.Q_eval.loss(q_target_steer, q_eval_steer).to(self.Q_eval.device)
        loss_acc = self.Q_eval.loss(q_target_acc, q_eval_acc).to(self.Q_eval.device)
        loss_brake = self.Q_eval.loss(q_target_brake, q_eval_brake).to(self.Q_eval.device)

        loss_brake.backward(retain_graph=True)
        loss_steer.backward(retain_graph=True)
        loss_acc.backward()

        self.Q_eval.optimizer.step()

        self.iter_counter += 1
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min







