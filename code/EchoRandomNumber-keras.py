# -*- coding: utf-8 -*-#
'''
# Name:         EchoRandomNumber-keras
# Description:  
# Author:       super
# Date:         2020/7/6
'''

from MiniFramework.DataReader_2_0 import *

train_file = "../data/ch19.train_echo.npz"
test_file = "../data/ch19.test_echo.npz"


def load_data():
    dr = DataReader_2_0(train_file, test_file)
    dr.ReadData()
    dr.GenerateValidationSet(k=10)
    
    return dr