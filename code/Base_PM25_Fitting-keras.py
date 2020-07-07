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

from keras.models import Sequential, load_model
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


def show_result(model, X, Y, num_step, pred_step, start, end):
    assert(X.shape[0] == Y.shape[0])
    count = X.shape[0] - X.shape[0] % pred_step
    A = np.zeros((count,1))

    for i in range(0, count, pred_step):
        A[i:i+pred_step] = predict(model, X[i:i+pred_step], num_step, pred_step)

    print(A.shape)
    print(Y.shape)
    plt.plot(A[start+1:end+1], 'r-x', label="Pred")
    plt.plot(Y[start:end], 'b-o', label="True")
    plt.legend()
    plt.show()

def predict(net, X, num_step, pred_step):
    A = np.zeros((pred_step, 1))
    for i in range(pred_step):
        x = set_predicated_value(X[i:i+1], A, num_step, i)
        a = net.predict(x)
        A[i,0] = a
    #endfor
    return A

def set_predicated_value(X, A, num_step, predicated_step):
    x = X.copy()
    for i in range(predicated_step):
        x[0, num_step - predicated_step + i, 0] = A[i]
    #endfor
    return x


if __name__ == '__main__':
    net_type = NetType.Fitting
    num_step = 24
    x_train, y_train, x_test, y_test, x_val, y_val = load_data(net_type, num_step)
    # print(x_train.shape)
    # print(y_train.shape)
    # print(x_test.shape)
    # print(y_test.shape)
    # print(x_val.shape)
    # print(y_train.shape)

    model = build_model()
    history = model.fit(x_train, y_train,
                        epochs=10,
                        batch_size=64,
                        validation_data=(x_val, y_val))
    # model = load_model("pm25.h5")
    print(model.summary())
    model.save("pm25.h5")
    draw_train_history(history)

    loss = model.evaluate(x_test, y_test)
    print("test loss: {}".format(loss))

    # pred_steps = [8,4,2,1]
    # for i in range(4):
    #     show_result(model, x_test, y_test, num_step, pred_steps[i], 1050, 1150)