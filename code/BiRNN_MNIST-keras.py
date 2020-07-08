# -*- coding: utf-8 -*-#
'''
# Name:         BiRNN_MNIST-keras
# Description:  
# Author:       super
# Date:         2020/7/8
'''

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import matplotlib.pyplot as plt

from ExtendedDataReader.MnistImageDataReader import *

from keras.models import Sequential, load_model
from keras.layers import SimpleRNN, Dense, Bidirectional



def load_data():
    dataReader = MnistImageDataReader(mode="timestep")
    dataReader.ReadData()
    dataReader.NormalizeX()
    dataReader.NormalizeY(NetType.MultipleClassifier, base=0)
    dataReader.Shuffle()
    dataReader.GenerateValidationSet(k=12)
    x_train, y_train = dataReader.XTrain, dataReader.YTrain
    x_test, y_test = dataReader.XTest, dataReader.YTest
    x_val, y_val = dataReader.XDev, dataReader.YDev
    x_train = x_train.squeeze()
    x_test = x_test.squeeze()
    x_val = x_val.squeeze()

    x_test_raw = dataReader.XTestRaw[0:64]
    y_test_raw = dataReader.YTestRaw[0:64]

    return x_train, y_train, x_test, y_test, x_val, y_val, x_test_raw, y_test_raw


def build_model():
    model = Sequential()
    model.add(Bidirectional(SimpleRNN(units=8), input_shape=(28,28)))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
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


def show_result(x, y, y_raw):
    x = x / 255
    fig, ax = plt.subplots(nrows=8, ncols=8, figsize=(11, 11))
    for i in range(64):
        ax[i // 8, i % 8].imshow(x[i].transpose(1, 2, 0).squeeze())
        print(y[i, 0])
        print(y_raw[i, 0])
        if y[i, 0] == y_raw[i, 0]:
            ax[i // 8, i % 8].set_title(y[i, 0])
        else:
            ax[i // 8, i % 8].set_title(y[i, 0], fontdict={'color':'r'})
        ax[i // 8, i % 8].axis('off')
    # endfor
    plt.show()


if __name__ == '__main__':
    x_train, y_train, x_test, y_test, x_val, y_val, x_test_raw, y_test_raw = load_data()
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    print(x_val.shape)
    print(y_train.shape)

    model = build_model()
    history = model.fit(x_train, y_train,
                        epochs=10,
                        batch_size=64,
                        validation_data=(x_val, y_val))
    model.save("BiRNN_MNIST.h5")
    print(model.summary())
    draw_train_history(history)

    loss, accuracy = model.evaluate(x_test, y_test)
    print("test loss: {}, test accuracy: {}".format(loss, accuracy))

    z = model.predict(x_test[0:64])
    show_result(x_test_raw[0:64], np.argmax(z, axis=1).reshape(64, 1), y_test_raw[0:64])