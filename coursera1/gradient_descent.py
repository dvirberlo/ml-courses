from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def linear_gradient_descent(X: NDArray, y: NDArray, alpha: float, iterations: int) -> Tuple[NDArray, float]:
    assert len(np.shape(X)) == 2
    assert len(np.shape(y)) == 1
    assert len(y) == np.shape(X)[0]
    assert alpha > 0
    assert iterations > 0
    w = np.zeros(np.shape(X)[1])
    b = 0
    for i in range(iterations):
        w, b = _linear_gradient_descent(w, b, X, y, alpha, i)
    return w, b


def _linear_gradient_descent(w: NDArray, b: float, X: NDArray, y: NDArray, alpha: float, iteration: int) -> Tuple[NDArray, float]:
    training_samples_count, features_count = X.shape
    y_prediction = np.dot(X, w) + b

    # dw = (1 / training_samples_count)
    dw = np.zeros(features_count)
    # for f in range(features_count):
    #     dw = np.dot(X[:, f], (y_prediction - y)) / training_samples_count
    #     w[f] -= alpha * dw
    # the following is vectorized version of the above loop
    dw = np.dot(X.T, (y_prediction - y)) / training_samples_count
    nw = w - alpha * dw

    db = np.sum(y_prediction - y) / training_samples_count
    nb = b - alpha * db

    w = nw
    b = nb

    # predicted_y = np.dot(X, nw) + nb
    # error = np.mean(np.square(predicted_y - y))

    return w, b


def _gradient_control(alpha: float, iteration_count: int, error_delta: np.floating) -> Tuple[bool, float]:
    # print(
    #     f"alpha: {alpha}, iteration_count: {iteration_count}, error_delta: {error_delta}")
    # if error_delta < 0:
    #     return False, alpha * 0.3
    # return True, alpha * 1.01
    return True, alpha


def main():
    _2d_data = [
        [[0], 1.3],
        [[1], 2.6],
        [[3], 7.9],
        [[5], 11],
    ]

    x: NDArray = np.array([row[0] for row in _2d_data])
    y: NDArray = np.array([row[1] for row in _2d_data])

    w, b = linear_gradient_descent(x, y, 0.1, 10)
    print(f"w: {w}, b: {b}")
    plt.scatter(x=[row[0] for row in x], y=y)
    plt.plot(x, np.dot(x, w) + b, color="red")
    plt.title("Linear Regression")
    plt.show()


if __name__ == "__main__":
    main()
