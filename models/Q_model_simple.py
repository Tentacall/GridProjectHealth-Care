import torch.nn as nn
import torch
import qsimcirq
from .utils import display
from .Quanv2D import Quanv2D

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    # print('CUDA is not available. Training on CPU...')
    device = torch.device('cpu')
else:
    # print('CUDA is available. Training on GPU...')
    device = torch.device('cuda:0')
    
    
class SimpleQClassifier(nn.Module):
    def __init__(self,device, testing = False, ):
        super().__init__()
        self.testing = testing
        self.device = device
        self.simulator = qsimcirq.QSimSimulator()
        self.quanv2d = nn.Sequential(
            Quanv2D(self.simulator, in_channels = 1, out_channels = 8, kernel_size = 2, stride = 1),
            Quanv2D(self.simulator, in_channels = 8, out_channels = 16, kernel_size = 2, stride = 1),
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, ceil_mode=False, dilation = 1,),
            Quanv2D(self.simulator, in_channels = 16, out_channels = 16, kernel_size = 2, stride = 1),
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, ceil_mode=False, dilation = 1,),
            # nn.AdaptiveMaxPool2d((5,5)),
        )
        
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=576, out_features=32*5*5, bias=True),
            nn.SiLU(inplace = True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=32*5*5, out_features=100, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=100, out_features=5, bias=True),
            # nn.Softmax()
        )
    
    def forward(self,  x):
        x = self.quanv2d(x)
        if self.testing: 
            display(x, "images","quantam_exp", True, True)
        print(x.shape)
        
        x = x.to(self.device)
        x = self.linear(x)
        print("Done")
        return x