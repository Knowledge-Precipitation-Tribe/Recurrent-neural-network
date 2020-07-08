# -*- coding: utf-8 -*-#
'''
# Name:         NameClassifier-keras
# Description:  
# Author:       super
# Date:         2020/7/8
'''

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import SimpleRNN, Masking

from ExtendedDataReader.NameDataReader import *

file = "../data/ch19.name_language.txt"


def load_data():
    dataReader = NameDataReader()
    dataReader.ReadData(file)
    dataReader.GenerateValidationSet(1000)
    x_train, y_train = dataReader.X, dataReader.Y
    x_val, y_val = dataReader.dev_x, dataReader.dev_y
    x_train = np.array(x_train)
    y_train= np.array(y_train)
    x_val = np.array(x_val)
    y_val= np.array(y_val)
    return x_train, y_train, x_val, y_val


def build_model():
    model = Sequential()
    model.add(SimpleRNN(input_shape=(4,2),
                        units=4))
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


if __name__ == '__main__':
    x_train, y_train, x_val, y_val = load_data()
    print(x_train.shape)
    print(y_train.shape)
    print(x_val.shape)

    # model = build_model()
    # history = model.fit(x_train, y_train,
    #                     epochs=1000,
    #                     batch_size=64,
    #                     validation_data=(x_val, y_val))
    # print(model.summary())
    # draw_train_history(history)
    #
    # loss, accuracy = model.evaluate(x_test, y_test)
    # print("test loss: {}, test accuracy: {}".format(loss, accuracy))