# -*- coding: utf-8 -*-#
'''
# Name:         EchoRandomNumber-keras
# Description:  
# Author:       super
# Date:         2020/7/6
'''

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import matplotlib.pyplot as plt

from MiniFramework.DataReader_2_0 import *

from keras.models import Sequential
from keras.layers import SimpleRNN

train_file = "../data/ch19.train_echo.npz"
test_file = "../data/ch19.test_echo.npz"


def load_data():
    dataReader = DataReader_2_0(train_file, test_file)
    dataReader.ReadData()
    dataReader.GenerateValidationSet(k=10)
    x_train, y_train = dataReader.XTrain, dataReader.YTrain
    x_test, y_test = dataReader.XTest, dataReader.YTest
    x_val, y_val = dataReader.XDev, dataReader.YDev
    return x_train, y_train, x_test, y_test, x_val, y_val


def build_model():
    model = Sequential()
    model.add(SimpleRNN(input_shape=(2,1),
                        units=2))
    model.compile(optimizer='Adam',
                  loss='mean_squared_error')
    return model

#画出训练过程中训练和验证的精度与损失
def draw_train_history(history):
    plt.figure(1)
    # summarize history for accuracy
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    plt.show()


if __name__ == '__main__':
    x_train, y_train, x_test, y_test, x_val, y_val = load_data()
    print(x_train.shape)
    print(x_test.shape)
    print(x_val.shape)

    model = build_model()
    history = model.fit(x_train, y_train,
                        epochs=50,
                        batch_size=64,
                        validation_data=(x_val, y_val))
    model.save("simplernn.h5")
    print(model.summary())
    draw_train_history(history)

    loss = model.evaluate(x_test, y_test)
    print("test loss: {}".format(loss))

