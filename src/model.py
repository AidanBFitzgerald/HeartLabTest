#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from tensorflow.keras import regularizers


def cnn_xray(loss='categorical_crossentropy', opt='adam'):

    # building a linear stack of layers with the sequential model
    model = Sequential()
    # convolutional layer
    model.add(
        Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu', input_shape=(256, 256, 1), kernel_regularizer= regularizers.l2(0.001), bias_regularizer= regularizers.l2(0.001)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    # flatten output of conv
    model.add(Flatten())
    # output layer
    model.add(Dense(3, activation='softmax'))

    # compiling the sequential model
    model.compile(loss=loss, metrics=['accuracy'], optimizer=opt)

    return model
