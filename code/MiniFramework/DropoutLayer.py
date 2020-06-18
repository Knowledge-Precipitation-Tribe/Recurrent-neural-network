# -*- coding: utf-8 -*-#
'''
# Name:         DropoutLayer
# Description:  
# Author:       super
# Date:         2020/6/12
'''

from MiniFramework.EnumDef_6_0 import *
from MiniFramework.Layer import *


class DropoutLayer(CLayer):
    def __init__(self, input_size, ratio=0.5):
        self.dropout_ratio = ratio  # the bigger, the more unit dropped
        self.mask = None
        self.input_size = input_size
        self.output_size = input_size

    def forward(self, input, train=True):
        assert (input.ndim == 2)
        if train:
            self.mask = np.random.rand(*input.shape) > self.dropout_ratio
            self.z = input * self.mask
        else:
            self.z = input * (1.0 - self.dropout_ratio)

        return self.z

    def backward(self, delta_in, idx):
        delta_out = self.mask * delta_in
        return delta_out