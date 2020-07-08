# -*- coding: utf-8 -*-#
'''
# Name:         LSTM_BinaryNumberMinus-keras
# Description:  
# Author:       super
# Date:         2020/7/8
'''

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import matplotlib.pyplot as plt

from MiniFramework.DataReader_2_0 import *

from keras.models import Sequential
from keras.layers import LSTM, Dense

train_file = "../data/ch19.train_minus.npz"
test_file = "../data/ch19.test_minus.npz"


def load_data():
    dataReader = DataReader_2_0(train_file, test_file)
    dataReader.ReadData()
    dataReader.Shuffle()
    dataReader.GenerateValidationSet(k=10)
    x_train, y_train = dataReader.XTrain, dataReader.YTrain
    x_test, y_test = dataReader.XTest, dataReader.YTest
    x_val, y_val = dataReader.XDev, dataReader.YDev
    return x_train, y_train, x_test, y_test, x_val, y_val


def build_model():
    model = Sequential()
    model.add(LSTM(input_shape=(4,2),
                        units=4))
    model.add(Dense(4, activation='sigmoid'))
    model.compile(optimizer='Adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

#画出训练过程中训练和验证的精度与损失
def draw_train_history(history):
    plt.figure(1)

    # summarize history for accuracy
    plt.subplot(211)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])

    # summarize history for loss
    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    plt.show()


def test(x_test, y_test, model):
    print("testing...")
    count = x_test.shape[0]
    result = model.predict(x_test)
    r = np.random.randint(0, count, 10)
    for i in range(10):
        idx = r[i]
        x1 = x_test[idx, :, 0]
        x2 = x_test[idx, :, 1]
        print("  x1:", reverse(x1))
        print("- x2:", reverse(x2))
        print("------------------")
        print("true:", reverse(y_test[idx]))
        print("pred:", reverse(result[idx]))
        x1_dec = int("".join(map(str, reverse(x1))), 2)
        x2_dec = int("".join(map(str, reverse(x2))), 2)
        print("{0} - {1} = {2}".format(x1_dec, x2_dec, (x1_dec - x2_dec)))
        print("====================")
    # end for


def reverse(a):
    l = a.tolist()
    l.reverse()
    return l


if __name__ == '__main__':
    x_train, y_train, x_test, y_test, x_val, y_val = load_data()
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(x_val.shape)

    model = build_model()
    history = model.fit(x_train, y_train,
                        epochs=200,
                        batch_size=64,
                        validation_data=(x_val, y_val))
    print(model.summary())
    draw_train_history(history)

    loss, accuracy = model.evaluate(x_test, y_test)
    print("test loss: {}, test accuracy: {}".format(loss, accuracy))

    test(x_test, y_test, model)