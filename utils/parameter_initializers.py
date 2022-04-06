import numpy as np


def initialize_with_zeros(train_set, number_of_nodes):
    w = np.zeros((train_set.shape[0], number_of_nodes))
    b = np.zeros((number_of_nodes, 1))
    return w, b


def initialize_with_random(x, number_of_nodes):
    w = np.random.randn(x.shape[0], number_of_nodes) * 0.01
    b = np.zeros((number_of_nodes, 1))
    return w, b
