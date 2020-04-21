import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, l1_dims, l2_dims, output_dims):
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
        self.output_dims = output_dims

        self.l1 = nn.Linear(*self.input_dims, self.l1_dims)
        self.l2 = nn.Linear(self.l1_dims, self.l2_dims)
        self.l3 = nn.Linear(self.l2_dims, output_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc1(x)

        return actions


