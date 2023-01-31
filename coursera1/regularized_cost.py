from abc import ABC, abstractmethod
from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


class FunctionForm(ABC):
    @abstractmethod
    def z(self, w: NDArray, b: float, X: NDArray) -> NDArray:
        pass

    @abstractmethod
    def w_shape(self, X: NDArray) -> Tuple:
        pass


class FunctionFormWithBoundary(FunctionForm):
    @abstractmethod
    def boundary(self, w: NDArray, b: float, X: NDArray) -> NDArray:
        pass


class LinearPrediction(FunctionForm):
    def z(self, w: NDArray, b: float, X: NDArray) -> NDArray:
        return np.dot(X, w) + b

    def w_shape(self, X: NDArray) -> Tuple:
        return (np.shape(X)[1],)


class LinearDecision2D(LinearPrediction, FunctionFormWithBoundary):
    def boundary(self, w: NDArray, b: float, X: NDArray) -> NDArray:
        return (w[0] * X + b) / (-w[1])


class SecondOrderPolynomial(FunctionForm):
    def z(self, w: NDArray, b: float, X: NDArray) -> NDArray:
        # y = np.zeros(len(X))
        # y = w[0] * X.T + w[1] * X.T ** 2 + b
        # for i in range(len(X)):
        #     print(i)
        #     print(X[i])
        #     print(w[0] * X[i])
        #     print(w[1] * X[i] ** 2)
        #     # y[i] = w[0] * X[i] + w[1] * X[i] ** 2 + b
        #     y[i] = (w[0] * X[i])[0] + (w[1] * X[i] ** 2)[0] + b
        # return y
        # the optimized version of the above loop:
        return (w[0] * X + w[1] * X ** 2 + b).flatten()[:len(X)]
        # return w[0] * X + w[1] * X ** 2 + b

    def w_shape(self, X: NDArray) -> Tuple:
        return (2,)


class SecondOrderDecision2D(SecondOrderPolynomial, FunctionFormWithBoundary):
    def boundary(self, w: NDArray, b: float, X: NDArray) -> NDArray:
        pass


def sigmoid(X: NDArray) -> NDArray:
    return 1 / (1 + np.exp(-X))


def learn(X: NDArray, y: NDArray, alpha: float, gamma: float, iterations: int, z: Callable[[NDArray, float, NDArray], NDArray], w_shape: Tuple, classification: bool = False) -> Tuple[NDArray, float]:
    assert len(np.shape(X)) == 2
    assert len(np.shape(y)) == 1
    assert len(y) == np.shape(X)[0]
    assert alpha > 0
    assert iterations > 0

    w = np.zeros(w_shape)
    b = 0
    for i in range(iterations):
        w, b = _learn(w, b, X, y, alpha, gamma, i, z, classification)
    return w, b


def _learn(w: NDArray, b: float, X: NDArray, y: NDArray, alpha: float, gamma: float, iteration: int, z: Callable[[NDArray, float, NDArray], NDArray], classification: bool) -> Tuple[NDArray, float]:
    # TODO: for now, ignores gamma
    training_samples_count, features_count = X.shape

    y_prediction = sigmoid(z(w, b, X)) if classification else z(w, b, X)
    # print(y_prediction)

    # dw = np.zeros(features_count)
    # for f in range(features_count):
    #     dw = np.dot(X[:, f], (y_prediction - y)) / training_samples_count
    #     w[f] -= alpha * dw
    # the following line is numpy's vectorized version of the above loop
    delta_w = np.dot(X.T, (y_prediction - y)) / training_samples_count
    # print(delta_w)
    # print(w)

    delta_b = np.sum(y_prediction - y) / training_samples_count

    w = w - alpha * delta_w
    # print(w)
    b = b - alpha * delta_b

    # y_prediction = sigmoid(z(w, b, X)) if classification else z(w, b, X)
    # error = np.mean(np.square(y_prediction - y))

    return w, b


def fluid_plot(X: NDArray, w: NDArray, b: float, y_func: Callable[[NDArray, float, NDArray], NDArray], num: int = 100) -> Tuple[NDArray, NDArray]:
    fluid_X = np.linspace(X.min(), X.max(), num)
    return fluid_X, y_func(w, b, fluid_X)


def main():
    # _decision_data = [
    #     [[0.1, 2.2], 0],
    #     [[0.2, 1.2], 0],
    #     [[0.3, 0.7], 0],
    #     [[1.3, 1.7], 0],
    #     [[1.7, 1.2], 0],
    #     [[0.5, 1.3], 0],
    #     [[1.2, 1.3], 0],
    #     [[0.4, 0.1], 0],
    #     [[0.2, 0.4], 0],
    #     [[0.9, 0.8], 0],
    #     [[1.2, 0.5], 0],
    #     [[2, 0.1], 0],

    #     [[1.2, 3.1], 1],
    #     [[1.4, 2.5], 1],
    #     [[2.2, 2.8], 1],
    #     [[2.8, 2.2], 1],
    #     [[2.4, 2.6], 1],
    #     [[2.5, 2.2], 1],
    #     [[3.2, 1], 1],
    #     [[3.3, 1.5], 1],
    #     [[3.4, 1.1], 1],
    #     [[3, 3], 1],
    # ]

    # X: NDArray = np.array([row[0] for row in _decision_data])
    # y: NDArray = np.array([row[1] for row in _decision_data])
    # alpha = 0.1
    # gamma = 0.1
    # iterations = 1000
    # # form = SecondOrderDecision2D()
    # form = LinearDecision2D()
    # w, b = learn(X, y, alpha, gamma, iterations, form.z,
    #              form.w_shape(X), classification=True)
    # print(f"w: {w}, b: {b}")
    # print(f"f(x) = {w[0]/(-w[1])}x + {b/(-w[1])}")
    # plt.scatter(x=[row[0] for row in X], y=[row[1] for row in X], c=y)
    # plt.plot(*fluid_plot(X[:, 0], w, b, form.boundary), color="red")

    _data = [
        [[0], 1.3],
        [[1], 2.6],
        [[3], 7.9],
        [[5], 11],
    ]
    # _data = [
    #     [[0], 0.3],
    #     [[1], 1.6],
    #     [[2], 4.9],
    #     [[4], 15],
    # ]
    # form = LinearPrediction()
    # X: NDArray = np.array([row[0] for row in _data])
    # y: NDArray = np.array([row[1] for row in _data])
    # w, b = learn(X, y, 0.01, 0.1, 1000, form.z, form.w_shape(X))
    # print(f"w: {w}, b: {b}")
    # plt.scatter(x=[row[0] for row in X], y=y)
    # plt.plot(X, np.dot(X, w) + b, color="red")
    # plt.title("Linear Regression")
    # plt.show()

    form = SecondOrderPolynomial()
    X: NDArray = np.array([row[0] for row in _data])
    y: NDArray = np.array([row[1] for row in _data])
    w, b = learn(X, y, 0.01, 0.1, 1000, form.z, form.w_shape(X))
    print(f"w: {w}, b: {b}")
    plt.scatter(x=[row[0] for row in X], y=y)
    plt.plot(*fluid_plot(X, w, b, form.z), color="red")
    plt.title("Second Order Regression")
    plt.show()


if __name__ == '__main__':
    main()

"""
X = [
    [0.1, 2.2],
    [0.2, 1.2],
    [0.3, 0.7],
    [1.3, 1.7],
]
w = [0. 0.]
b = 0.
=>
y = [
    [0.]
    [0.]
    [0.]
    [0.]
]
=> sigmoid(y) = [
    [0.5]
    [0.5]
    [0.5]
    [0.5]
]


"""
