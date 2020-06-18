# -*- coding: utf-8 -*-#
'''
# Name:         Layer
# Description:  
# Author:       super
# Date:         2020/6/12
'''

class CLayer(object):
    def __init__(self, layer_type):
        self.layer_type = layer_type

    def initialize(self, folder, name):
        pass

    def train(self, input, train=True):
        pass

    def pre_update(self):
        pass

    def update(self):
        pass

    def save_parameters(self):
        pass

    def load_parameters(self):
        pass