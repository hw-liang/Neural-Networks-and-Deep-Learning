from __future__ import print_function
from data_processor_r import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
"""
torch.nn是专门为神经网络设计的模块化接口。nn构建于autograd之上，可以用来定义和运行神经网络。
nn.Module是nn中十分重要的类,包含网络各层的定义及forward方法;Module既可以表示神经网络中的某个层（layer），也可以表示一个包含很多层的神经网络。
定义自已的网络:
  需要继承nn.Module类，并实现forward方法。
  一般把网络中具有可学习参数的层放在构造函数__init__()中,不具有可学习参数的层(如ReLU)可放在构造函数中，
也可不放在构造函数中(而在forward中使用nn.functional来代替)
  只要在nn.Module的子类中定义了forward函数，backward函数就会被自动实现(利用Autograd). 在forward函数中可以使用任何Variable支持的函数，
因为在整个pytorch构建的图中，是Variable在流动。还可以使用if,for,print,log等python语法.        
注：Pytorch基于nn.Module构建的模型中，只支持mini-batch的Variable输入方式
"""
class MyConv2d(nn.Module):
    """
    Our simplified implemented of nn.Conv2d module for 2D convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding=None):
        super(MyConv2d, self).__init__()  # 自定义层必须继承nn.Module，并且在其构造函数中需调用nn.Module的构造函数

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if padding is None:
            self.padding = kernel_size // 2
        else:
            self.padding = padding
        self.weight = nn.parameter.Parameter(torch.Tensor(
            out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.parameter.Parameter(torch.Tensor(out_channels))
        # parameter是一种特殊的Variable，但其默认需要求导（requires_grad = True）
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels * self.kernel_size * self.kernel_size
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, padding=self.padding)

class RegressionCNN(nn.Module):
    def __init__(self, kernel, num_filters):
        # first call parent's initialization function
        super(RegressionCNN, self).__init__()
        padding = kernel // 2

        self.downconv1 = nn.Sequential(
            nn.Conv2d(1, num_filters, kernel_size=kernel, padding=padding),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.MaxPool2d(2),)
        self.downconv2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters*2, kernel_size=kernel, padding=padding),
            nn.BatchNorm2d(num_filters*2),
            nn.ReLU(),
            nn.MaxPool2d(2),)

        self.rfconv = nn.Sequential(
            nn.Conv2d(num_filters*2, num_filters*2, kernel_size=kernel, padding=padding),
            nn.BatchNorm2d(num_filters*2),
            nn.ReLU(),)

        self.upconv1 = nn.Sequential(
            nn.Conv2d(num_filters*2, num_filters, kernel_size=kernel, padding=padding),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),)
        self.upconv2 = nn.Sequential(
            nn.Conv2d(num_filters, 3, kernel_size=kernel, padding=padding),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),)
        self.finalconv = MyConv2d(3, 3, kernel_size=kernel)

    def forward(self, x):
        out = self.downconv1(x)
        out = self.downconv2(out)
        out = self.rfconv(out)
        out = self.upconv1(out)
        out = self.upconv2(out)
        out = self.finalconv(out)
        # 在前向传播函数中，我们有意识地将输出变量都命名成out，是为了能让Python回收一些中间层的输出，从而节省内存。但并不是所有都会被回收，
        # 有些variable虽然名字被覆盖，但其在反向传播仍需要用到，此时Python的内存回收模块将通过检查引用计数，不会回收这一部分内存。
        # 返回值也是一个Variable对象
        return out