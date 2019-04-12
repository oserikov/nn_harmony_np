import numpy as np


def softmax(x):
    z = x - np.max(x)
    sm = (np.exp(z).T / np.sum(np.exp(z), axis=0)).T
    return sm


def deriv_tanh(tanh_value):
    return 1 - tanh_value * tanh_value


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def deriv_sigmoid(sigmoid_value):
    return sigmoid_value * (1. - sigmoid_value)


def relu(x):
    return np.maximum(x, 0)


def deriv_relu(relu_value):
    return 1. * (relu_value > 0)


def neg_log(x):
    # TODO: does this actually work?
    return -np.log(x)
