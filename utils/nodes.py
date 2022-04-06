import numpy as np


def logistic_regression(x: np.ndarray, w: np.ndarray, b: int):
    return np.dot(w.T,  x) + b


def sigmoid(z: np.ndarray):
    return 1 / (1 + np.exp(-z))


# def sigmoid_derivative(z: np.ndarray):
#     ds = sigmoid(z) * (1 - sigmoid(z))
#     return ds


def sigmoid_derivative(z: np.ndarray):
    ds = np.divide(np.exp(-z), (1 + np.exp(-z)) ** 2)
    return ds


def tanh(z: np.ndarray):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))


def tanh_derivative(z: np.ndarray):
    return 1 - (tanh(z)) ** 2


def softmax(x: np.ndarray):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    s = np.divide(x_exp, x_sum)
    return s


def loss_function(y, y_hat):
    return - (y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


def cost_function(y, y_hat):
    m_examples = y.shape[1]
    cost = -1 / m_examples * np.sum((y, * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))

    return cost


