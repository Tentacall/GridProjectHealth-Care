import torch.nn as nn
import torch
import qsimcirq
from .utils import display, loading_bar
from .Quanv2D import Quanv2D

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    # print('CUDA is not available. Training on CPU...')
    device = torch.device('cpu')
else:
    # print('CUDA is available. Training on GPU...')
    device = torch.device('cuda:0')

class QClassifier(nn.Module):
    def __init__(self, testing = False):
        super().__init__()
        self.testing = testing
        self.simulator = qsimcirq.QSimSimulator()
        self.classical_expansion = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size= (2,2), stride = (1,1), padding = (1,1)),
            nn.SiLU(inplace = True),
            nn.Conv2d(64, 128, kernel_size= (2,2), stride = (1,1), padding = (1,1)),
            nn.LPPool2d(kernel_size = 2, stride = 2,  ceil_mode=False, norm_type = 2),
            
            nn.Conv2d(128, 32, kernel_size= (2,2), stride = (1,1), padding = (1,1)),
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, ceil_mode=False, dilation = 1,)
        )
        
        self.quantam_expansion = nn.Sequential(
            Quanv2D(self.simulator, in_channels = 32, out_channels = 32, kernel_size = 2, stride = 1),
            nn.AdaptiveMaxPool2d((20,20)),
            Quanv2D(self.simulator, in_channels = 32, out_channels = 32, kernel_size = 2, stride = 1),
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, ceil_mode=False, dilation = 1,)
        )
        
        self.fully_connected = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=32*9*9, out_features=4096, bias=True),
            nn.SiLU(inplace = True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=1000, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1000, out_features=5, bias=True),
        )

        
    def forward(self, x):
        x = self.classical_expansion(x)
        print("Classical Expansion: Done")
        # if self.testing: 
            # display(x, "images","classical_expansion", True, True)
            
        x = self.quantam_expansion(x)
        print("Quantam Expansion Done")
        # if self.testing: 
            # display(x, "images","quantam_expansion", True, True)
        # print(x)
        x = x.to(device)
        x = self.fully_connected(x)
        print("Done")
        return x