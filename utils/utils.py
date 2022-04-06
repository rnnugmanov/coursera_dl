import copy
import matplotlib.pyplot as plt

import numpy as np
from coursera_dl.utils.nodes import sigmoid, logistic_regression


def propagate(w, b, X, Y):
    m_examples = X.shape[1]

    A = sigmoid(logistic_regression(X, w, b))
    cost = -1 / m_examples * np.sum((Y * np.log(A) + (1 - Y) * np.log(1 - A)))

    dw = (1 / m_examples) * np.dot(X, (A - Y).T)
    db = np.float64((1 / m_examples) * np.sum((A - Y)))

    cost = np.squeeze(np.array(cost))
    grads = {'dw': dw, 'db': db}

    return grads, cost


def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)

    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads['dw']
        db = grads['db']

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)

            if print_cost:
                print("Cost after iteration %i: %f" % (i, cost))

    params = {'w': w, 'b': b}
    grads = {'dw': dw, 'db': db}

    return params, grads, costs


def predict(w, b, X):
    m_examples = X.shape[1]
    w = w.reshape(X.shape[0], 1)
    predictions = np.zeros((1, m_examples))

    A = sigmoid(logistic_regression(X, w, b))

    for i in range(A.shape[1]):
        if A[0, i] > 0.5:
            predictions[0, i] = 1
        else:
            predictions[0, i] = 0

    return predictions
