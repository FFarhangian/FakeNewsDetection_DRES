# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
# License: BSD 3-Clause License
#
# Adapted from: https://github.com/rafaelmenelau/instance-hardness
#
# Reference:
# Smith, M.R., Martinez, T., & Giraud-Carrier, C. (2014).
# "An instance level analysis of data complexity."
# Machine Learning, 95(2), 225–256. https://doi.org/10.1007/s10994-013-5377-2

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.stats import mode

def hardness_region_competence(neighbors_idx, labels, safe_k):
    """
    Calculate instance hardness based on neighborhood class overlap.

    Parameters
    ----------
    neighbors_idx : array-like of shape [n_samples_test, k]
        Indices of the nearest neighbors for each sample.

    labels : array-like of shape [n_samples_train]
        Class labels of the training data.

    safe_k : int
        Number of neighbors used in the region of competence.

    Returns
    -------
    hardness : ndarray of shape [n_samples_test]
        Instance hardness scores.

    Notes
    -----
    This method is based on:
    Smith, M.R., Martinez, T., & Giraud-Carrier, C. (2014).
    "An instance level analysis of data complexity."
    Machine Learning, 95(2), 225–256.
    """
    if neighbors_idx.ndim < 2:
        neighbors_idx = np.atleast_2d(neighbors_idx)

    neighbors_y = labels[neighbors_idx[:, :safe_k]]
    _, num_majority_class = mode(neighbors_y, axis=1, keepdims=False)
    hardness = ((safe_k - num_majority_class) / safe_k).reshape(-1, )
    return hardness

def kdn_score(X, y, k):
    """
    Compute the K-Disagreeing Neighbors (KDN) score for each instance.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature vectors.

    y : array-like of shape (n_samples,)
        Corresponding class labels.

    k : int
        Number of neighbors.

    Returns
    -------
    score : ndarray of shape (n_samples,)
        KDN scores for each instance.

    neighbors : ndarray of shape (n_samples, k)
        Indices of k nearest neighbors.

    References
    ----------
    Smith, M.R., Martinez, T., & Giraud-Carrier, C. (2014).
    "An instance level analysis of data complexity."
    Machine Learning, 95(2), 225–256.
    """
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='kd_tree').fit(X)
    _, indices = nbrs.kneighbors(X)
    neighbors = indices[:, 1:]  # exclude self
    diff_class = np.tile(y, (k, 1)).T != y[neighbors]
    score = np.sum(diff_class, axis=1) / k
    return score, neighbors
