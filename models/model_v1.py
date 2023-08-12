import torch.nn as nn
from .qcModel import Quanv2D
import torch

class ClassicalModel(nn.Module):
    def __init__(self):
        super(ClassicalModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 12, kernel_size = 2, stride = 1 )
        self.conv2 = nn.Conv2d(in_channels = 12, out_channels = 48, kernel_size = 2, stride = 1 )
        self.conv = nn.Conv2d(in_channels = 48, out_channels = 48, kernel_size = 2, stride = 1)
        self.fc1 = nn.Linear(48*6*6, 50)
        self.fc2 = nn.Linear(50, 5)
        self.maxpool = nn.MaxPool2d(2, stride = 2)
        self.minpool = nn.AvgPool2d(2, stride = 2)
        self.activation = nn.SiLU()
        
    def forward(self, x):
        # print(x.size())
        x = self.activation(self.maxpool(self.conv1(x))) # 12x111x111
        # print(x.size())
        x = torch.relu(self.maxpool(self.conv2(x))) # 48x55x55
        # print(x.size())
        x = self.activation(self.maxpool(self.conv(x))) # 48x27x27
        # print(x.size())
        x = torch.relu(self.maxpool(self.conv(x))) # 48x13x13
        # print(x.size())
        x = self.activation(self.maxpool(self.conv(x))) # 48x6x6
        # print(x.size())
        x = x.view(-1, 48*6*6)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x