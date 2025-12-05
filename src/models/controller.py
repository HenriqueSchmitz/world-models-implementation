import torch

class Controller(torch.nn.Module):
    def __init__(self, observation_dim, hidden_dim, action_dim):
        super(Controller, self).__init__()
        self.fc = torch.nn.Linear(observation_dim + hidden_dim, action_dim)

    def forward(self, observation, hidden_state):
        x = torch.cat([observation, hidden_state], dim=1)
        x = self.fc(x)
        steering = torch.tanh(x[:, 0:1])
        gas = torch.sigmoid(x[:, 1:2])
        brake = torch.sigmoid(x[:, 2:3])
        return torch.cat([steering, gas, brake], dim=1)