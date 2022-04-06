import copy

import numpy as np
import matplotlib.pyplot as plt

from coursera_dl.utils.datasets_create import load_planar_dataset, load_extra_datasets
from coursera_dl.utils.plots import plot_decision_boundary, plot_cost


# # Plot the decision boundary for logistic regression
# plot_decision_boundary(lambda x: clf.predict(x), X, Y)
# plt.title("Logistic Regression")
#
# # Print accuracy
# LR_predictions = clf.predict(X.T)
# print('Accuracy of logistic regression: %d ' % float(
#     (np.dot(Y, LR_predictions) + np.dot(1 - Y, 1 - LR_predictions)) / float(Y.size) * 100) +
#       '% ' + "(percentage of correctly labelled datapoints)")


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def matmul(w, x, b):
    return np.dot(w, x) + b


def initialize_with_random(x_train, layers: list):
    layers.insert(0, x_train.shape[0])
    parameters = {}
    for layer in range(0, len(layers) - 1):
        parameters['W{}'.format(layer + 1)] = np.random.randn(layers[layer + 1], layers[layer]) * 0.01
        parameters['b{}'.format(layer + 1)] = np.zeros((layers[layer + 1], 1))

    return parameters


def forward_propagation(x_train, parameters):
    Z1 = matmul(parameters['W1'], x_train, parameters['b1'])
    A1 = np.tanh(Z1)
    Z2 = matmul(parameters['W2'], A1, parameters['b2'])
    A2 = sigmoid(Z2)

    cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}

    return cache


def backward_propagation(X, Y, parameters, cache):
    m = X.shape[1]

    W1 = parameters['W1']
    W2 = parameters['W2']

    A1 = cache['A1']
    A2 = cache['A2']

    dZ2 = A2 - Y
    dW2 = (1 / m) * (np.dot(dZ2, A1.T))
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    # YOUR CODE ENDS HERE

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads


def update_parameters(parameters, grads, learning_rate):
    parameters = copy.deepcopy(parameters)
    for i in range(len(parameters) // 2):
        parameters['W{}'.format(i + 1)] = parameters['W{}'.format(i + 1)] - learning_rate * grads['dW{}'.format(i + 1)]
        parameters['b{}'.format(i + 1)] = parameters['b{}'.format(i + 1)] - learning_rate * grads['db{}'.format(i + 1)]

    return parameters


def predict(x, parameters):
    cache = forward_propagation(x, parameters)
    predictions_hat = cache['A2'] > 0.5

    return predictions_hat


def compute_cost(A2, y_train):
    m_examples = y_train.shape[1]
    return -1 / m_examples * np.sum((y_train * np.log(A2) + (1 - y_train) * np.log(1 - A2)))


def model(x_train, y_train, x_test, y_test, layers: list, num_iterations=2000, learning_rate=1.2):
    parameters = initialize_with_random(x_train, layers)

    costs = []
    for i in range(num_iterations):
        cache = forward_propagation(x_train, parameters)
        cost = compute_cost(cache['A2'], y_train)
        grads = backward_propagation(x_train, y_train, parameters, cache)
        parameters = update_parameters(parameters, grads, learning_rate)

        if i % 100 == 0:
            costs.append(cost)
            print("Cost after iteration %i: %f" % (i, cost))

    train_predictions = predict(x_train, parameters)
    print('Accuracy for train set: %d' % float(
        (np.dot(y_train, train_predictions.T) + np.dot(1 - y_train, 1 - train_predictions.T)) / float(
            y_train.size) * 100) + '%')

    # plot_decision_boundary(lambda x: predict(x_train, parameters), x_train, y_train)
    test_predictions = predict(x_test, parameters)
    print('Accuracy for test set: %d' % float(
        (np.dot(y_test, test_predictions.T) + np.dot(1 - y_test, 1 - test_predictions.T)) / float(
            y_test.size) * 100) + '%')

    return {'costs': costs, 'learning_rate': learning_rate}


if '__main__' == __name__:
    # 1. set up model structure
    layers_structure = [4, 1]

    # 2. prepare dataset
    # train_set_x, train_set_y = load_planar_dataset()
    examples = 200
    train_examples = 150
    train_set_x_orig, train_set_y_orig = load_extra_datasets('gaussian_quantiles', examples)

    train_set_x_orig = train_set_x_orig.reshape(-1, examples)
    train_set_y_orig = train_set_y_orig.reshape(-1, examples)

    train_set_x = train_set_x_orig[:, :train_examples]
    test_set_x = train_set_x_orig[:, train_examples:]

    train_set_y = train_set_y_orig[:, :train_examples]
    test_set_y = train_set_y_orig[:, train_examples:]

    # 2.5 (optional) show data
    # plt.scatter(train_set_x[0, :], train_set_x[1, :], c=train_set_y[0], s=40, cmap=plt.cm.Spectral)
    # plt.show()

    # 3. train and infer
    one_hidden_layer_model = model(train_set_x, train_set_y, test_set_x, test_set_y, layers_structure,
                                   num_iterations=10000, learning_rate=1.2)

    # 4. plot costs
    plot_cost(one_hidden_layer_model['costs'], one_hidden_layer_model['learning_rate'])
