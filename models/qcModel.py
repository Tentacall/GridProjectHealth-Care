import torch.nn as nn
from .circuits.ConvKernel import kernel2, kernel3
import torch

class Quanv2D(nn.Module):
    def __init__(self, simulator, in_channels = 1, out_channels = 1, kernel_size = 2, stride = 1, precision = 10):
        super().__init__()
        self.simulator = simulator
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.kernel = kernel3
        self.precision = precision
        self.extension_factor = self.out_channels // self.in_channels
        self.weight = nn.Parameter(torch.rand((out_channels, self.kernel_size*self.kernel_size)))
        print(self.weight)

    def patches_generator(self, image):
        image_height, image_width = image.size()
        output_height = ( image_height - self.kernel_size )// self.stride +1
        output_width = ( image_width - self.kernel_size )//self.stride + 1
        
        for h in range(output_height):
            for w in range(output_width):
                patch = image[h*self.stride:(h*self.stride+self.kernel_size), w*self.stride:(w*self.stride+self.kernel_size)]
                patch = patch.reshape(self.kernel_size*self.kernel_size).tolist()
                yield patch, h, w
    
    def forward(self, x):
        kernel_height, kernel_width = (2,2)
        count, in_channels, image_height, image_width = x.size()
        expansion = self.in_channels*4 == self.out_channels
        output_height = ( image_height - kernel_height )// self.stride +1
        output_weidth = ( image_width - kernel_width )//self.stride + 1
        result = torch.zeros(count, self.out_channels, output_height, output_weidth)
        for c in range(count):
            print(f"[c] : {c}")
            for ic in range(in_channels):
                for patch, h, w in self.patches_generator(x[c][ic]):
                    for t in range(self.extension_factor):
                        W = [w.item() for w in self.weight[ic*self.extension_factor + t]]
                        circuit, keys = self.kernel(patch, W)
                        res = self.simulator.run(circuit, repetitions=self.precision)
                        result[c][ic*self.extension_factor + t][h][w] = res.histogram(key = "q")[1]*0.1
        return result

    def backward(self):
        pass
