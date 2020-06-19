# -*- coding: utf-8 -*-#
'''
# Name:         EnumDef_6_0
# Description:  
# Author:       super
# Date:         2020/6/12
'''

import numpy as np
#import minpy.numpy as np
#from minpy.context import set_context, gpu

from enum import Enum

#set_context(gpu(0))

class NetType(Enum):
    Fitting = 1,
    BinaryClassifier = 2,
    MultipleClassifier = 3

class InitialMethod(Enum):
    Zero = 0,
    Normal = 1,
    Xavier = 2,
    MSRA = 3

class XCoordinate(Enum):
    Nothing = 0,
    Iteration = 1,
    Epoch = 2

class OptimizerName(Enum):
    SGD = 0,
    Momentum = 1,
    Nag = 2,
    AdaGrad = 3,
    AdaDelta = 4,
    RMSProp = 5,
    Adam = 6

class StopCondition(Enum):
    Nothing = 0,    # reach the max_epoch then stop
    StopLoss = 1,   # reach specified loss value then stop
    StopDiff = 2,   # reach specified abs(curr_loss - prev_loss)

class Stopper(object):
    def __init__(self, sc, sv):
        self.stop_condition = sc
        self.stop_value = sv

class RegularMethod(Enum):
    Nothing = 0,
    L1 = 1,
    L2 = 2,
    EarlyStop = 3

class PoolingTypes(Enum):
    MAX = 0,
    MEAN = 1,

class ParameterType(Enum):
    Init = 0,
    Best = 1,
    Last = 2

class OutputType(Enum):
    EachStep = 1,
    LastStep = 2