import tensorflow as tf
import tensorflow.keras
from pathlib import Path
import numpy as np

TRANSPOSE = False

def normalize(image):
    image = tf.cast(image, tf.float32) / 255.
    return image


def get_dataset():
    import h5py

    train_dataset = h5py.File(Path(__file__).parent.parent / 'dataset' / 'train_signs.h5', "r")
    test_dataset = h5py.File(Path(__file__).parent.parent / 'dataset' / 'test_signs.h5', "r")

    x_train_set = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_x'])
    y_train_set = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_y'])

    x_test_set = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_x'])
    y_test_set = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_y'])

    new_x_train = x_train_set.map(normalize)
    new_x_test = x_test_set.map(normalize)

    return new_x_train, y_train_set, new_x_test, y_test_set


def initialize_parameters_he(layers, X):
    layers.insert(0, np.prod(X.element_spec.shape))
    initializer = tf.keras.initializers.GlorotNormal(seed=1)
    parameters = {}
    for i in range(1, len(layers)):
        W_shape = [layers[i], layers[i - 1]] if TRANSPOSE else [layers[i - 1], layers[i]]
        b_shape = [layers[i], 1] if TRANSPOSE else [1, layers[i]]
        parameters['W' + str(i)] = tf.Variable(initializer(shape=W_shape))
        parameters['b' + str(i)] = tf.Variable(initializer(shape=b_shape))
    return parameters


def forward_prop(X, parameters):
    A_prev = tf.reshape(X, [X.shape[0], -1])

    if TRANSPOSE:
        A_prev = tf.transpose(A_prev)
    mm = lambda a, b: tf.linalg.matmul(b, a) if TRANSPOSE else tf.linalg.matmul(a, b)

    A_prev = tf.keras.activations.relu(
        tf.math.add(
            mm(A_prev, parameters['W' + str(1)]), parameters['b' + str(1)]))
    A_prev = tf.keras.activations.relu(
        tf.math.add(
            mm(A_prev, parameters['W' + str(2)]), parameters['b' + str(2)]))

    A_prev = tf.math.add(mm(A_prev, parameters['W' + str(3)]), parameters['b' + str(3)])

    if TRANSPOSE:
        A_prev = tf.transpose(A_prev)
    return A_prev


def compute_cost(X, Y):
    cost = tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(Y, X, from_logits=True))
    return cost


def model(X, Y, X_test, Y_test, layer_structure, learning_rate=0.0001, num_epochs=1000000, mini_batch_size=32,
          print_cost=True):
    parameters = initialize_parameters_he(layer_structure, X)

    dataset = tf.data.Dataset.zip((X, Y))
    dataset_test = tf.data.Dataset.zip((X_test, Y_test))

    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

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
                ZL = forward_prop(minibatch_x, parameters)
                cost = compute_cost(ZL, minibatch_y)

            grads = tape.gradient(cost, parameters)
            optimizer.apply_gradients(zip(grads.values(), parameters.values()))
            train_accuracy.update_state(minibatch_y, ZL)

            epoch_cost += cost

        epoch_cost /= m

        if print_cost is True and epoch % 10 == 0:
            print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            print("Train accuracy:", train_accuracy.result())

            # We evaluate the test set every 10 epochs to avoid computational overhead
            for (minibatch_x, minibatch_y) in test_minibatches:
                ZL = forward_prop(minibatch_x, parameters)
                test_accuracy.update_state(minibatch_y, ZL)
                if test_accuracy.result().numpy() > 0.85:
                    np.savez(r'C:\Users\RuslanNN\PycharmProjects\learning\coursera_dl\mock_models\parameters.npz',
                             **parameters)
            print("Test_accuracy:", test_accuracy.result())

            costs.append(epoch_cost)
            train_acc.append(train_accuracy.result())
            test_acc.append(test_accuracy.result())
            test_accuracy.reset_states()

    if test_acc[-1].numpy() > 0.85:
        np.savez(r'C:\Users\RuslanNN\PycharmProjects\learning\coursera_dl\mock_models\parameters.npz', **parameters)
    return parameters, costs, train_acc, test_acc


if '__main__' == __name__:
    layer_dims = [2, 3, 6]
    x_train, y_train, x_test, y_test = get_dataset()
    parameters, costs, train_accuracy, test_accuracy = model(x_train, y_train, x_test, y_test, layer_dims)
    print(train_accuracy, test_accuracy)
