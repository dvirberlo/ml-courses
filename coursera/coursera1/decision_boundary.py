from abc import ABC, abstractmethod
from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


class FunctionForm(ABC):
    @abstractmethod
    def z(self, w: NDArray, b: float, x: NDArray) -> NDArray:
        pass

    @abstractmethod
    def boundary(self, x: float, w: NDArray, b: float) -> float:
        pass


class FunctionFormLinear2D(FunctionForm):
    def z(self, w: NDArray, b: float, x: NDArray) -> NDArray:
        return np.dot(x, w) + b

    def boundary(self, x: float, w: NDArray, b: float) -> float:
        return (w[0] * x + b) / (-w[1])


def decision_boundary(X: NDArray, y: NDArray, alpha: float, iterations: int, z: Callable[[NDArray, float, NDArray], NDArray]) -> Tuple[NDArray, float]:
    assert len(np.shape(X)) == 2
    assert len(np.shape(y)) == 1
    assert len(y) == np.shape(X)[0]
    assert alpha > 0
    assert iterations > 0
    w = np.zeros(np.shape(X)[1])
    b = 0
    for i in range(iterations):
        w, b = _decision_boundary(w, b, X, y, alpha, i, z)
    return _decision_boundary(w, b, X, y, alpha, iterations, z)


def _decision_boundary(w: NDArray, b: float, X: NDArray, y: NDArray, alpha: float, iteration: int, z: Callable[[NDArray, float, NDArray], NDArray]) -> Tuple[NDArray, float]:
    training_samples_count, features_count = np.shape(X)
    y_prediction = sigmoid(z(w, b, X))

    # for f in range(features_count):
    #     dw = np.dot(X[:, f], (y_prediction - y)) / training_samples_count
    #     w[f] -= alpha * dw
    # the following is vectorized version of the above loop
    dw = np.dot(X.T, (y_prediction - y)) / training_samples_count
    nw = w - alpha * dw

    db = (1 / training_samples_count) * np.sum(y_prediction - y)
    nb = b - alpha * db

    w = nw
    b = nb

    # y_pred = sigmoid(np.dot(x, nw) + nb)
    # error = np.mean(np.square(y_pred - y))

    return w, b


def sigmoid(X: NDArray) -> NDArray:
    return 1 / (1 + np.exp(-X))

# def _get_2d_plot(w: NDArray, b: float, X: NDArray) -> Tuple[list, list]:
#     return [row[0] for row in X], [form.boundary(row[0]) for row in X]


def _binary_classification(X: NDArray, y: NDArray, alpha: float, iterations: int, form: FunctionForm) -> Tuple[NDArray, float, Tuple[list, list]]:
    w, b = decision_boundary(X, y, alpha, iterations, form.z)
    plot = [row[0] for row in X], [form.boundary(row[0], w, b) for row in X]
    return w, b, plot


def main():
    _data = [
        [[0.1, 2.2], 0],
        [[0.2, 1.2], 0],
        [[0.3, 0.7], 0],
        [[1.3, 1.7], 0],
        [[1.7, 1.2], 0],
        [[0.5, 1.3], 0],
        [[1.2, 1.3], 0],
        [[0.4, 0.1], 0],
        [[0.2, 0.4], 0],
        [[0.9, 0.8], 0],
        [[1.2, 0.5], 0],
        [[2, 0.1], 0],

        [[1.2, 3.1], 1],
        [[1.4, 2.5], 1],
        [[2.2, 2.8], 1],
        [[2.8, 2.2], 1],
        [[2.4, 2.6], 1],
        [[2.5, 2.2], 1],
        [[3.2, 1], 1],
        [[3.3, 1.5], 1],
        [[3.4, 1.1], 1],
        [[3, 3], 1],
    ]

    X: NDArray = np.array([row[0] for row in _data])
    y: NDArray = np.array([row[1] for row in _data])
    alpha = 0.1
    iterations = 1000
    form = FunctionFormLinear2D()

    w, b, boundary_line = _binary_classification(X, y, alpha, iterations, form)
    print(f"w: {w}, b: {b}")
    print(f"f(x) = {w[0]/(-w[1])}x + {b/(-w[1])}")

    plt.scatter(x=[row[0] for row in X], y=[row[1]
                for row in X], c=y, cmap="bwr")
    # plt.scatter(x=[row[0] for row in X], y=[row[1]
    #             for row in X], c=[form.z(w, b, row) for row in X])

    plt.plot(*boundary_line)
    plt.title("Decision Boundary")
    plt.show()


if __name__ == "__main__":
    main()
