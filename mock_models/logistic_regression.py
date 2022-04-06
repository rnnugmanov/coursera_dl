import copy

from coursera_dl.utils.datasets_create import load_dataset
from coursera_dl.utils.nodes import sigmoid
from coursera_dl.utils.parameter_initializers import initialize_with_random
import numpy as np


def initialize_with_zeros(train_set, number_of_nodes):
    w = np.zeros((train_set.shape[0], number_of_nodes))
    b = np.zeros((number_of_nodes, 1))
    return w, b


def propagate(w, b, X, Y):
    m_examples = X.shape[1]

    A = sigmoid(np.dot(w.T, X) + b)
    cost = -1 / m_examples * np.sum((Y * np.log(A) + (1 - Y) * np.log(1 - A)))

    dw = (1 / m_examples) * np.dot(X, (A - Y).T)
    db = np.float64((1 / m_examples) * np.sum((A - Y)))

    cost = np.squeeze(np.array(cost))

    grads = {'dw': dw, 'db': db}

    return grads, cost


def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.005):
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    dw, db = None, None

    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads['dw']
        db = grads['db']

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)
            print("Cost after iteration %i: %f" % (i, cost))

    params = {'w': w, 'b': b}
    grads = {'dw': dw, 'db': db}

    return params, grads, costs


def predict(w, b, X):
    y_prediction = np.zeros((1, X.shape[1]))
    A = sigmoid(np.dot(w.T, X) + b)

    for example in range(A.shape[1]):
        if A[0, example] > 0.5:
            y_prediction[0, example] = 1
        else:
            y_prediction[0, example] = 0

    return y_prediction


def model(train_set_X, train_set_Y, test_set_X, test_set_Y, nodes, num_iterations=2000, learning_rate=0.005):
    w, b, = initialize_with_random(train_set_X, nodes)
    params, grads, costs = optimize(w, b, train_set_X, train_set_Y,
                                    learning_rate=learning_rate, num_iterations=num_iterations)
    w, b = params['w'], params['b']

    train_predict = predict(w, b, train_set_X)
    test_predict = predict(w, b, test_set_X)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(train_predict - train_set_Y)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(test_predict - test_set_Y)) * 100))


if '__main__' == __name__:
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset(r'dataset/train_catvnoncat.h5',
                                                                                   r'dataset/test_catvnoncat.h5')
    train_set_x_flattened = train_set_x_orig.reshape(-1, train_set_x_orig.shape[0])
    test_set_x_flattened = test_set_x_orig.reshape(-1, test_set_x_orig.shape[0])
    train_set_x = train_set_x_flattened / 255.
    test_set_x = test_set_x_flattened / 255.

    nodes = 5
    model(train_set_x, train_set_y, test_set_x, test_set_y, nodes, num_iterations=50000)
