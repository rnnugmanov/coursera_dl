import h5py
import numpy as np
import tensorflow.keras
import tensorflow as tf


def normalize(image):
    image = tf.cast(image, tf.float32) / 255.
    image = tf.reshape(image, [-1])
    return image


def one_hot_matrix(labels, depth=6):
    one_hot = tf.reshape(tf.one_hot(labels, depth, axis=0), [-1])
    return one_hot


def get_dataset():
    train_dataset = h5py.File(r'C:\Users\RuslanNN\PycharmProjects\learning\coursera_dl\dataset\train_signs.h5', "r")
    test_dataset = h5py.File(r'C:\Users\RuslanNN\PycharmProjects\learning\coursera_dl\dataset\test_signs.h5', "r")

    x_train_set = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_x'])
    y_train_set = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_y'])

    x_test_set = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_x'])
    y_test_set = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_y'])

    new_x_train = x_train_set.map(normalize)
    new_x_test = x_test_set.map(normalize)

    return new_x_train, y_train_set, new_x_test, y_test_set


def initialize_parameters_he(layers, X):
    layers.insert(0, X.element_spec.shape[0])
    initializer = tf.keras.initializers.GlorotNormal(seed=1)
    parameters = {}
    for i in range(1, len(layers)):
        parameters['W' + str(i)] = tf.Variable(initializer(shape=[layers[i], layers[i - 1]]))
        parameters['b' + str(i)] = tf.Variable(initializer(shape=[layers[i], 1]))

    return parameters


def forward_prop(X, parameters):
    L = len(parameters) // 2
    A_prev = X
    Z = None

    for l in range(1, L + 1):
        Z = tf.math.add(tf.linalg.matmul(parameters['W' + str(l)], A_prev), parameters['b' + str(l)])
        if l == L:
            return Z
        else:
            A_prev = tf.keras.activations.relu(Z)

    return Z


def compute_cost(X, Y):
    cost = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(tf.transpose(Y), tf.transpose(X), from_logits=True))
    return cost


def model(X, Y, X_test, Y_test, layer_structure, learning_rate=0.0001, num_epochs=10000, mini_batch_size=32,
          print_cost=True):
    parameters = initialize_parameters_he(layer_structure, X)

    Y = Y.map(one_hot_matrix)
    Y_test = Y_test.map(one_hot_matrix)

    dataset = tf.data.Dataset.zip((X, Y))
    dataset_test = tf.data.Dataset.zip((X_test, Y_test))

    train_accuracy = tf.keras.metrics.CategoricalAccuracy()
    test_accuracy = tf.keras.metrics.CategoricalAccuracy()

    m = dataset.cardinality().numpy()

    optimizer = tf.keras.optimizers.Adam(learning_rate)

    minibatches = dataset.batch(mini_batch_size).prefetch(8)
    test_minibatches = dataset_test.batch(mini_batch_size).prefetch(8)

    costs = []
    train_acc = []
    test_acc = []

    for epoch in range(num_epochs):
        epoch_cost = 0.
        train_accuracy.reset_states()

        for (minibatch_x, minibatch_y) in minibatches:
            with tf.GradientTape() as tape:
                ZL = forward_prop(tf.transpose(minibatch_x), parameters)
                cost = compute_cost(ZL, tf.transpose(minibatch_y))

            grads = tape.gradient(cost, parameters)
            optimizer.apply_gradients(zip(grads.values(), parameters.values()))
            train_accuracy.update_state(minibatch_y, tf.transpose(ZL))

            epoch_cost += cost

        epoch_cost /= m

        if print_cost is True and epoch % 10 == 0:
            print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            print("Train accuracy:", train_accuracy.result())

            # We evaluate the test set every 10 epochs to avoid computational overhead
            for (minibatch_x, minibatch_y) in test_minibatches:
                ZL = forward_prop(tf.transpose(minibatch_x), parameters)
                test_accuracy.update_state(minibatch_y, tf.transpose(ZL))
            print("Test_accuracy:", test_accuracy.result())

            costs.append(epoch_cost)
            train_acc.append(train_accuracy.result())
            test_acc.append(test_accuracy.result())
            test_accuracy.reset_states()

    return parameters, costs, train_acc, test_acc


if '__main__' == __name__:
    layer_dims = [2, 3, 6]
    x_train, y_train, x_test, y_test = get_dataset()
    parameters, costs, train_accuracy, test_accuracy = model(x_train, y_train, x_test, y_test, layer_dims)
    print(train_accuracy, test_accuracy)

