import torch.nn as nn
import gymnasium as gym


class DQN_Model(nn.Module):
    def __init__(self, input_shape, action_space):
        super(DQN_Model, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=8, stride=4)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=8)

        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, action_space)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = nn.functional.tanh(self.fc1(x))
        x = self.fc2(x)
        return x
