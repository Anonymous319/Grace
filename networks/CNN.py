import sys
import torch
import torch.nn as nn
import numpy as np


class Net(nn.Module):
    def __init__(self, inputsize, nclass):
        super().__init__()
        channel = inputsize[0]
        im_size = (inputsize[1],inputsize[2])
        net_width, net_depth = 128, 3
        self.net_act = nn.LeakyReLU(negative_slope=0.01) 
        self.net_pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, im_size)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
        self.last=torch.nn.Linear(num_feat,nclass)


    def _make_layers(self, channel, net_width, net_depth, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            layers += [nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)]
            layers += [self.net_act]
            in_channels = net_width
            layers += [self.net_pooling]
            shape_feat[1] //= 2
            shape_feat[2] //= 2
        return nn.Sequential(*layers), shape_feat

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        y = self.last(out)
        return y



class Net2(nn.Module):
    def __init__(self, inputsize, taskcla):
        super().__init__()
        channel = inputsize[0]
        im_size = (inputsize[1],inputsize[2])
        net_width, net_depth = 128, 3
        self.net_act = nn.LeakyReLU(negative_slope=0.01)
        self.taskcla = taskcla
        self.net_pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, im_size)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]

        self.last=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(num_feat,n))
 
    def _make_layers(self, channel, net_width, net_depth, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            layers += [nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)]
            layers += [self.net_act]
            in_channels = net_width
            layers += [self.net_pooling]
            shape_feat[1] //= 2
            shape_feat[2] //= 2
        return nn.Sequential(*layers), shape_feat

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        y = []
        for t,i in self.taskcla:
            y.append(self.last[t](out))
        return y

    def forward_hid(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out
