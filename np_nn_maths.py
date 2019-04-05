import numpy as np


def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z), axis=0)).T
    return sm


def deriv_tanh(tanh_value):
    return 1 - tanh_value * tanh_value


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def deriv_sigmoid(sigmoid):
    return sigmoid * (1. - sigmoid)


def neg_log(x):
    # TODO: does this actually work?
    return -np.log(x)