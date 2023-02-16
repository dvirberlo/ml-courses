from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.markers import MarkerStyle
from numpy.typing import NDArray


def k_means(data: NDArray, k: int) -> Tuple[NDArray, NDArray]:
    """
    :param data: array of shape (m, d)
    :param k: number of clusters
    :return: centroids, labels
    """
    m, d = data.shape
    centroids = data[np.random.choice(m, k, replace=False)]
    labels = np.zeros(m, dtype=int)
    while True:
        # _distances = [
        #     [
        #         np.linalg.norm(data[i] - centroids[j])
        #         for j in range(k)
        #     ]
        #     for i in range(m)
        # ]
        # this is the vectorized version of the above
        _distances = np.linalg.norm(data[:, None] - centroids, axis=2)
        new_labels = np.argmin(_distances, axis=1)
        if np.all(labels == new_labels):
            break
        labels = new_labels
        _to_remove: list[int] = []
        for i in range(len(centroids)):
            _assigned = data[labels == i]
            if len(_assigned) == 0:
                _to_remove.append(i)
            else:
                centroids[i] = _assigned.mean(axis=0)
        if _to_remove:
            centroids = np.delete(centroids, _to_remove)

    return centroids, labels


def many_k_means(data: NDArray, k: int, n: int) -> Tuple[NDArray, NDArray]:
    """
    :param data: array of shape (m, d)
    :param k: number of clusters
    :param n: number of times to run k_means
    :return: centroids, labels
    """
    m, d = data.shape
    centroids = np.zeros((k, d))
    best_labels = np.zeros(m, dtype=int)
    best_cost = np.inf
    for _ in range(n):
        centroids, labels = k_means(data, k)
        cost = 0
        for i in range(len(centroids)):
            cost += np.linalg.norm(data[labels == i] - centroids[i])
        if cost < best_cost:
            best_cost = cost
            best_labels = labels
    return centroids, best_labels


def main():
    _2d_data = [
        [[0], 1],
        [[0], 2],
        [[1], 0],
        [[1], 4],
        [[2], 3],
        [[2], 2],

        [[3], 6],
        [[3.5], 6],
        [[4], 5],
        [[5], 3],
        [[5], 5],
        [[6], 4],
        [[6], 3],
        [[6], 6.5],
    ]

    X: NDArray = np.array([row[0] for row in _2d_data])
    y: NDArray = np.array([row[1] for row in _2d_data])

    centroids, labels = many_k_means(X, 2, 20)
    plt.scatter(x=[row[0] for row in X], y=y, c=labels)
    # draw centroids with X
    plt.scatter(x=[row[0] for row in centroids], y=[row[0]
                for row in centroids], c="red", marker=MarkerStyle("X"))
    plt.title("Clustering")
    plt.show()


if __name__ == "__main__":
    main()
