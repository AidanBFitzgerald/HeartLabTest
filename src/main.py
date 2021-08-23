#!/usr/bin/env python3

# Imports!
import os
import random
import time
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard

# Config and Inits
from HeartLabTest.src.model import cnn_xray

data_dir = "chest_xray"
size = (256, 256)


# Fucntions!
def img_2_arr(
        img_path: str,
        resize: bool = False,
        grayscale: bool = True,
        size: tuple = (256, 256),
) -> np.ndarray:
    """
    This function is responsible for opening an image, Preprocessing
    it by color or size and returning a numpy array.

    Input:
        - img_path: str, a path to the location of a image file on disk
        - resize: bool, True/False if the image is to be resized
        - grayscale: bool, True/False if image is meant to be B&W or color
        - size: tuple, a 2d tuple containing the x/y size of the image.

    Output:
        - a np.ndarray which is assosiated to the image that was input.
    """

    if grayscale:
        img_arr = cv2.imread(img_path, 0)
    else:
        img_arr = cv2.imread(img_path)

    if resize:
        img_arr = cv2.resize(img_arr, size)

    return img_arr


def create_datasets(data_dir: str) -> np.ndarray:
    """
    This function is responsible for creating a dataset which
    contains all images and their associated class.

    Inputs:
        - data_dir: str, which is the location where the chest x-rays are
            located.

    Outputs:
        - a np.ndarray which contains the processed image, and the class
            int, associated with that class.

    """
    # Image Loading and Preprocessing
    all_normal_img_paths = []
    all_viral_img_paths = []
    all_bact_img_paths = []
    for cls in os.listdir(data_dir):  # NORMAL or PNEUMONIA
        for img in os.listdir(os.path.join(data_dir, cls)):  # all images
            if cls == "NORMAL":
                all_normal_img_paths.append(os.path.join(data_dir, cls, img))
            elif "virus" in img:
                all_viral_img_paths.append(os.path.join(data_dir, cls, img))
            else:
                all_bact_img_paths.append(os.path.join(data_dir, cls, img))

    # 0 for normal, 1 for bacterial and 2 for viral
    dataset = (
            [
                [img_2_arr(path, grayscale=True, resize=True, size=size), 0]
                for path in all_normal_img_paths
            ]
            + [
                [img_2_arr(path, grayscale=True, resize=True, size=size), 1]
                for path in all_bact_img_paths
            ]
            + [
                [img_2_arr(path, grayscale=True, resize=True, size=size), 2]
                for path in all_viral_img_paths
            ]
    )

    return np.array(dataset, dtype="object")


def split_dataset(dataset):
    # shuffle dataset for better training
    random.shuffle(dataset)
    X = []
    y = []
    # Split dataset into images and labels
    for features, label in dataset:
        X.append(features)
        y.append(label)
    X = np.array(X)
    y = np.array(y)
    return X, y


def preprocess(X, y):
    X = X.reshape(-1, size[0], size[1], 1)
    # apply appropriate number of categories to y
    y = to_categorical(y, 3)
    # normalise the data
    X = X / 255.0

    return X, y



def main():
    # get dataset
    dataset = create_datasets(data_dir)
    model = cnn_xray()
    X, y = split_dataset(dataset)
    X, y = preprocess(X, y)

    NAME = "cnn_xray-{}".format(int(time.time()))
    tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

    model.fit(X, y, batch_size=20, epochs=10, validation_split=0.1, callbacks=[tensorboard])


if __name__ == "__main__":
    main()
