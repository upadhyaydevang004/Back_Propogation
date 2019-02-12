#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mlp.py
# Author: Qian Ge <qge2@ncsu.edu>

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

sys.path.append('../')
import src.network2 as network2
import src.mnist_loader as loader
import src.activation as act
from src.network2 import load

DATA_PATH = 'E:/SEM 2/Neural/project1/data'

[val_loss, val_acc, train_loss, train_acc] = [0, 0, 0, 0]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', action='store_true',
                        help='Check data loading.')
    parser.add_argument('--sigmoid', action='store_true',
                        help='Check implementation of sigmoid.')
    parser.add_argument('--gradient', action='store_true',
                        help='Gradient check')
    parser.add_argument('--train', action='store_true',
                        help='Train the model')

    return parser.parse_args()


def load_data():
    train_data, valid_data, test_data = loader.load_data_wrapper(DATA_PATH)
    print('Number of training: {}'.format(len(train_data[0])))
    print('Number of validation: {}'.format(len(valid_data[0])))
    print('Number of testing: {}'.format(len(test_data[0])))
    return train_data, valid_data, test_data


def test_sigmoid():
    z = np.arange(-10, 10, 0.1)
    y = act.sigmoid(z)
    y_p = act.sigmoid_prime(z)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(z, y)
    plt.title('sigmoid')

    plt.subplot(1, 2, 2)
    plt.plot(z, y_p)
    plt.title('derivative sigmoid')
    plt.show()


def gradient_check():
    train_data, valid_data, test_data = load_data()
    model = network2.Network([784, 300, 100, 10])
    model.gradient_check(training_data=train_data, layer_id=1, unit_id=5, weight_id=3)


def main():
    # load train_data, valid_data, test_data
    train_data, valid_data, test_data = load_data()
    # construct the network
    model = network2.Network([784, 300, 100, 10])
    # train the network using SGD

    num_epochs = 100
    lr = 3e-3

    [val_loss, val_acc, train_loss, train_acc] = model.SGD(
        training_data=train_data,
        epochs=num_epochs,
        mini_batch_size=128,
        eta=lr,
        lmbda=0.0,
        evaluation_data=valid_data,
        monitor_evaluation_cost=True,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True,
        monitor_training_accuracy=True)
    _predict(model, test_data)

    model.save('weights.json')

    plt.subplots_adjust(hspace=0.5)

    plt.subplot(2, 1, 1)
    plt.title("Loss (Eta = %.4f)" % lr)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    line1, = plt.plot(train_loss, label='Train Loss')
    line2, = plt.plot(val_loss, label='Validation Loss')
    plt.legend(handles=[line1, line2])
    plt.grid()

    train_acc = np.array(train_acc)
    val_acc = np.array(val_acc)

    train_acc = train_acc / 30.0
    val_acc = val_acc / 100.0

    plt.subplot(2, 1, 2)
    plt.title("Accuracy (Eta = %.4f)" % lr)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    line1, = plt.plot(train_acc, label='Train Accuracy')
    line2, = plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(handles=[line1, line2])
    plt.grid()
    plt.show()



def _predict(model,test_data):
    # predict the test labels and print accuracy score
    accuracy = model.accuracy(test_data)
    print("[testing accuracy]: {}%".format(round((accuracy / len(test_data[0])), 4) * 100))


if __name__ == '__main__':
    FLAGS = get_args()
    if FLAGS.input:
        load_data()
    if FLAGS.sigmoid:
        test_sigmoid()
    if FLAGS.train:
        main()
    if FLAGS.gradient:
        gradient_check()



