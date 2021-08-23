#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten



def model_zero(loss='categorical_crossentropy', opt='adam'):
    """
    This function implements a small example model that you can use/modify.
    You are not limited to creating your own model, you can use already
    implemented models such as ResNets etc.
    """
    # building a linear stack of layers with the sequential model
    model = Sequential()
    # convolutional layer
    model.add(Conv2D(25, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(28,28,1)))
    model.add(MaxPool2D(pool_size=(1,1)))
    # flatten output of conv
    model.add(Flatten())
    # hidden layer
    model.add(Dense(100, activation='relu'))
    # output layer
    model.add(Dense(10, activation='softmax'))

    # compiling the sequential model
    model.compile(loss=loss, metrics=['accuracy'], optimizer=opt)

    return model
