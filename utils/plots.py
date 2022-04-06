import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from coursera_dl.utils.nodes import sigmoid


def plot_cost(costs, learning_rate):
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()


def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y[0], cmap=plt.cm.Spectral)
    mpl.pyplot.show()


def make_plot(func: callable):
    data = np.linspace(-15, 15, 100)
    values = list(map(func, data))

    fig, axes = plt.subplots(figsize=(15, 15))
    axes.spines["left"].set_position(("data", 0))
    axes.spines["bottom"].set_position(("data", 0))
    axes.spines["top"].set_visible(True)
    axes.spines["right"].set_visible(True)

    axes.plot(data, values, label=func.__name__)

    axes.set_title('First plot')
    axes.legend()
    mpl.pyplot.show()


if '__main__' == __name__:
    make_plot(np.tanh)
