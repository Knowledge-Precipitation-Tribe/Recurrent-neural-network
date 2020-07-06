# -*- coding: utf-8 -*-#
'''
# Name:         Base_PM25_Fitting-keras
# Description:  
# Author:       super
# Date:         2020/7/6
'''

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import matplotlib.pyplot as plt

from ExtendedDataReader.PM25DataReader import *

from keras.models import Sequential
from keras.layers import SimpleRNN

train_file = "../data/ch19.train_echo.npz"
test_file = "../data/ch19.test_echo.npz"


def load_data(net_type, num_step):
    dataReader = PM25DataReader(net_type, num_step)
    dataReader.ReadData()
    dataReader.Normalize()
    dataReader.GenerateValidationSet(k=1000)
    x_train, y_train = dataReader.XTrain, dataReader.YTrain
    x_test, y_test = dataReader.XTest, dataReader.YTest
    x_val, y_val = dataReader.XDev, dataReader.YDev
    return x_train, y_train, x_test, y_test, x_val, y_val


def build_model():
    model = Sequential()
    model.add(SimpleRNN(input_shape=(24,6),
                        units=1))
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


def show_result(model, x_test, y_test, num_step, pred_step, start, end):
    count = x_test.shape[0] - x_test.shape[0] % pred_step
    A = np.zeros((count,1))

    for i in range(0, count, pred_step):
        A[i:i+pred_step] = model.predict(model, x_test[i:i+pred_step], num_step, pred_step)

    plt.plot(A[start+1:end+1], 'r-x', label="Pred")
    plt.plot(y_test[start:end], 'b-o', label="True")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    net_type = NetType.Fitting
    num_step = 24
    x_train, y_train, x_test, y_test, x_val, y_val = load_data(net_type, num_step)
    print(x_train.shape)
    print(x_test.shape)
    print(x_val.shape)
    print(y_train.shape)

    model = build_model()
    history = model.fit(x_train, y_train,
                        epochs=10,
                        batch_size=64,
                        validation_data=(x_val, y_val))
    print(model.summary())
    draw_train_history(history)

    loss = model.evaluate(x_test, y_test)
    print("test loss: {}".format(loss))

    pred_steps = [8, 4, 2, 1]
    for i in range(4):
        show_result(model, x_test, y_test, num_step, pred_steps[i], 1050, 1150)