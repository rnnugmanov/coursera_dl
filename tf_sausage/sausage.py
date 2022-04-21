from pathlib import Path

import h5py
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

from mock_models.tf_model import get_dataset, one_hot_matrix, compute_cost


def fc(layers_structure: list, flattened_shape: list):
    last_layer_index = len(layers_structure) - 1

    blocks = [tf.keras.layers.Flatten(input_shape=flattened_shape)]
    for i, units in enumerate(layers_structure):
        activation = 'relu' if i != last_layer_index else 'softmax'
        blocks.append(keras.layers.Dense(units, activation=activation))
    return blocks


def train(x_train, y_train, model: keras.Model,
          epoch_num: int, learning_rate: float, batch_size: int):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=1000)


def train_as_expert(X: tf.data.Dataset, Y: tf.data.Dataset, model: keras.Model,
                    epoch_num: int, learning_rate: float, batch_size: int):
    Y = Y.map(one_hot_matrix)
    dataset = tf.data.Dataset.zip((X, Y))
    m = dataset.cardinality().numpy()
    minibatches = dataset.batch(batch_size).prefetch(8)

    optimizer = tf.keras.optimizers.Adam(learning_rate)

    for epoch in range(epoch_num):
        epoch_cost = 0.
        for (minibatch_x, minibatch_y) in minibatches:
            with tf.GradientTape() as tape:
                logits = model(minibatch_x)
                loss_value = compute_cost(logits, minibatch_y)
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            epoch_cost += loss_value

        epoch_cost /= m

        if epoch % 10 == 0:
            print("Cost after epoch %i: %f" % (epoch, epoch_cost))


def get_clean_dataset():
    train_dataset = h5py.File(Path(__file__).parent.parent / 'dataset' / 'train_signs.h5', "r")
    test_dataset = h5py.File(Path(__file__).parent.parent / 'dataset' / 'test_signs.h5', "r")

    x_train_set = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_x'])
    y_train_set = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_y'])

    x_test_set = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_x'])
    y_test_set = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_y'])

    nf = lambda X: np.array(list(X.as_numpy_iterator()), dtype=np.float32)
    ni = lambda X: np.array(list(X.as_numpy_iterator()))

    return (nf(x_train_set), ni(y_train_set)), (nf(x_test_set), ni(y_test_set))

#     x_train, x_test = x_train / 255.0, x_test / 255.0
#
#     depth = np.max(y_train) + 1
#     blocks = fc([2, 3, depth], x_train.shape[1:])
#     model = keras.models.Sequential(blocks)
#
#     model.summary()
#
#     train(x_train, y_train, model, epoch_num=5000, learning_rate=0.0001, batch_size=32)
#
#     callback = keras.callbacks.ModelCheckpoint()
#     model.save(r'C:\Users\Unicorn\git\coursera_dl\tf_sausage', save_format='tf')


if __name__ == '__main__':
    print(tf.__version__)
    (x_train, y_train), (x_test, y_test) = get_clean_dataset()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=x_train.shape[1:]),
        keras.layers.Dense(2, activation='relu', kernel_initializer='glorot_normal'),
        keras.layers.Dense(3, activation='relu', kernel_initializer='glorot_normal'),
        keras.layers.Dense(6, activation='softmax', kernel_initializer='glorot_normal')
    ])

    model.compile(optimizer=keras.optimizers.Adam(0.0001),
                  # loss=lambda y_pred, y_true: tf.math.reduce_mean(keras.metrics.sparse_categorical_crossentropy(y_pred, y_true)),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    model.fit(x_train, y_train, epochs=5000, batch_size=128)
    model.evaluate(x_test, y_test)
# 1.7918