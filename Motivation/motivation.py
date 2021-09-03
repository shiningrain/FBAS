import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import autokeras as ak
from keras.datasets import mnist,cifar10
from keras.models import load_model
from keras import backend as K
import pandas
import keras
import sys
import pickle
import tensorflow as tf
import time
import argparse

def cifar10_load_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
        x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test image classification')
    parser.add_argument('-method','-m',default='auto', help='model path')# 'auto' 'cust'
    args = parser.parse_args()

    (x_train, y_train), (x_test, y_test) = cifar10_load_data()


    if args.method=='auto':
        clf = ak.ImageClassifier(
        overwrite=True,
        max_trials=10,tuner='bayesian')
        # Feed the image classifier with training data.
        clf.fit(x_train, y_train, epochs=100)

    print('finish')