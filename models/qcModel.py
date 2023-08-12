import torch.nn as nn
from .circuits.ConvKernel import kernel2
import torch

class Quanv2D(nn.Module):
    def __init__(self, simulator, in_channels = 1, out_channels = 1, kernel_size = 2, stride = 2, precision = 10):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(4))*100
        self.bias = nn.Parameter(torch.zeros(4))
        self.simulator = simulator
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.kernel = kernel2
        self.precision = precision
        assert self.in_channels == self.out_channels or self.in_channels*4 == self.out_channels

    def forward(self, x):
        kernel_height, kernel_width = (self.kernel_size, self.kernel_size)
        count, in_channels, image_height, image_width = x.size()
        expansion = self.in_channels*4 == self.out_channels
        result = torch.zeros(count, self.out_channels, image_height//2, image_width//2)
        for c in range(count):
            for ic in range(in_channels):
                for i in range(0, image_height - kernel_height + 1, self.stride ):
                    for j in range(0, image_width - kernel_width + 1, self.stride ):
                        P = x[c][ic][i:i+kernel_height][j:j+kernel_width]
                        P = [P[0][1], P[0][1], P[1][0], P[1][1]]
                        circuit, keys = self.kernel(P, self.weight)
                        res = self.simulator.run(circuit, self.precision)
                        if expansion:
                            for p in range(4):
                                result[c][ic*4 + p][i//2][j//2] = res.histogram(key = keys[p])[1]/self.precision
                        else:
                            result[c][ic][i//2][j//2] = res.histogram(key=keys[3])[1]/self.precision
        return result

    def backward(self):
        pass
