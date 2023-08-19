import torch.nn as nn
import torch
from .utils import display

class SimpleClassifier(nn.Module):
    def __init__(self, testing = False):
        super().__init__()
        self.features = nn.Sequential(
            
            nn.Conv2d(1, 64, kernel_size=(2,2), stride=(1, 1), padding=(1, 1)),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(2,2), stride=(1, 1), padding=(1, 1)),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            
            nn.Conv2d(64, 128, kernel_size=(2,2), stride=(1, 1), padding=(1, 1)),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(2,2), stride=(1, 1), padding=(1, 1)),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        )
        
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=128*8*8, out_features=100, bias=True),
            nn.SiLU(inplace = True),
            nn.Linear(in_features=100, out_features=5, bias=True),
            nn.Softmax(dim=1)
        )
    
    def forward(self,  x):
        x = self.features(x)
        # print(x.shape)
        
        x = self.linear(x)
        return x
        