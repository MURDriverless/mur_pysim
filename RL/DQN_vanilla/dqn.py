import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, l1_dims, l2_dims, steer_dims, acc_dims, brake_dims):
        """

        :param lr:
        :param input_dims:
        :param l1_dims:
        :param l2_dims:
        :param output_dims:
        """

        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.l1_dims = l1_dims
        self.l2_dims = l2_dims

        self.l1 = nn.Linear(*self.input_dims, self.l1_dims)
        self.l2 = nn.Linear(self.l1_dims, self.l2_dims)
        self.l3_steer = nn.Linear(self.l2_dims, steer_dims)
        self.l3_acc = nn.Linear(self.l2_dims, acc_dims)
        self.l3_brake = nn.Linear(self.l2_dims, brake_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        action_steer = self.l3_steer(x)
        action_acc = self.l3_acc(x)
        action_brake = self.l3_brake(x)

        return action_steer, action_acc, action_brake


