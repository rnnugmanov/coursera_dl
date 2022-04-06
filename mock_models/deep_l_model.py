import copy

import sklearn.datasets
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


from coursera_dl.working_models.deep_model import relu_backward, sigmoid_backward


def relu(Z):
    A = np.maximum(Z, 0)
    cache = Z
    return A, cache


def sigmoid(Z):
    cache = Z
    A = 1 / (1 + np.exp(-Z))
    return A, cache


def initialize_parameters_deep(X, num_layers):
    num_layers.insert(0, X.shape[0])
    parameters = {}

    for i in range(1, len(num_layers)):
        parameters['W' + str(i)] = np.random.randn(num_layers[i], num_layers[i - 1]) * 0.01
        parameters['b' + str(i)] = np.zeros((num_layers[i], 1))

    return parameters


def initialize_parameters_deep_he(X, num_layers):
    num_layers.insert(0, X.shape[0])
    parameters = {}

    for i in range(1, len(num_layers)):
        parameters['W' + str(i)] = np.random.randn(num_layers[i], num_layers[i - 1]) * np.sqrt(2 / num_layers[i - 1])
        parameters['b' + str(i)] = np.zeros((num_layers[i], 1))

    return parameters


def linear_forward(A, w, b):
    z = np.dot(w, A) + b
    cache = (A, w, b)
    return z, cache


def linear_activation_forward(A_prev, w, b, activation):
    A, activation_cache = None, None

    Z, linear_cache = linear_forward(A_prev, w, b)
    if activation == 'relu':
        A, activation_cache = relu(Z)
    if activation == 'sigmoid':
        A, activation_cache = sigmoid(Z)

    cache = (linear_cache, activation_cache)
    return A, cache


def l_deep_model_forward(A_prev, parameters):
    caches = []
    forward_parameters = copy.deepcopy(parameters)
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev, cache = linear_activation_forward(A_prev, forward_parameters['W' + str(l)],
                                                  forward_parameters['b' + str(l)], 'relu')
        caches.append(cache)

    AL, cache = linear_activation_forward(A_prev, forward_parameters['W' + str(L)], forward_parameters['b' + str(L)],
                                          'sigmoid')
    caches.append(cache)

    return AL, caches


def compute_cost(AL, Y):
    m_examples = Y.shape[1]
    cost = -1 / m_examples * (np.dot(Y, np.log(AL).T) + np.dot(1 - Y, np.log(1 - AL).T))
    cost = np.squeeze(cost)
    return cost


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1 / m * np.dot(dZ, A_prev.T)
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    dA_prev, dW, db, dZ = None, None, None, None

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)

    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def l_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)  # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[-1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, 'sigmoid')
    grads["dA" + str(L - 1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp

    # Loop from l=L-2 to l=0
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, 'relu')
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(1, L):
        parameters['W' + str(l)] = parameters['W' + str(l)] - learning_rate * grads['dW' + str(l)]
        parameters['b' + str(l)] = parameters['b' + str(l)] - learning_rate * grads['db' + str(l)]

    return parameters


def l_deep_model(X, Y, num_layers, weights_initialization, learning_rate=0.5, num_iterations=5000):
    assert weights_initialization in ['usual', 'he']
    if weights_initialization == 'usual':
        parameters = initialize_parameters_deep(X, num_layers)
    if weights_initialization == 'he':
        parameters = initialize_parameters_deep_he(X, num_layers)

    A_prev = X
    costs = []
    iterations = []

    for i in range(num_iterations):
        AL, caches = l_deep_model_forward(A_prev, parameters)
        cost = compute_cost(AL, Y)

        if i % 50 == 0:
            print('Cost value is: {}'.format(cost))
            costs.append(cost)
            iterations.append(i)

        grads = l_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

    return parameters, costs, iterations


def predict(X, Y, parameters):
    m_examples = X.shape[1]
    Y_hat, caches = l_deep_model_forward(X, parameters)

    for i in range(m_examples):
        if Y_hat[0, i] > 0.5:
            Y_hat[0, i] = 1
        else:
            Y_hat[0, i] = 0

    accuracy = np.sum((Y_hat == Y) / m_examples) * 100
    print("Accuracy: " + str(np.sum((Y_hat == Y) / m_examples) * 100) + "%")
    # if accuracy > 90:
    #     np.savez(r'C:\Users\RuslanNN\PycharmProjects\learning\coursera_dl\working_models\parameters.npz', **parameters)


def make_plot(scatters, iterations):
    plot, axes = plt.subplots(figsize=(5, 5))
    axes.spines["left"].set_position(("data", 0))
    axes.spines["bottom"].set_position(("data", 0))
    axes.spines["top"].set_visible(True)
    axes.spines["right"].set_visible(True)

    axes.plot(iterations, scatters, label='Cost function')

    axes.set_title('First plot')
    axes.legend()
    mpl.pyplot.show()


def run_model(layers: list, n_samples: int, train_set_examples: int, mode: str, show_plot: bool, parameters_path=None):
    assert mode in ['train', 'infer']
    parameters = {}
    dataset_x, dataset_y = sklearn.datasets.make_moons(n_samples=n_samples, noise=.3)
    dataset_y = np.reshape(dataset_y, (1, -1))

    train_x = dataset_x[:train_set_examples].T
    train_y = dataset_y[:, :train_set_examples]

    test_x = dataset_x[train_set_examples:].T
    test_y = dataset_y[:, train_set_examples:]

    if mode == 'train':
        parameters, costs, iterations = l_deep_model(train_x, train_y, layers, 'he',
                                                     learning_rate=1.80, num_iterations=3000)
        if show_plot:
            make_plot(costs, iterations)

    elif mode == 'infer':
        data = np.load(parameters_path)
        for k in data.files:
            parameters[k] = data[k]

    predict(train_x, train_y, parameters)
    predict(test_x, test_y, parameters)


if '__main__' == __name__:
    N = 500000
    train_set_size = int(N * 0.9)
    layers_set = [4, 2, 1]
    data_path = r'C:\Users\RuslanNN\PycharmProjects\learning\coursera_dl\working_models\parameters.npz'

    run_model(layers=layers_set, n_samples=N, train_set_examples=train_set_size, mode='infer', show_plot=False,
              parameters_path=data_path)
