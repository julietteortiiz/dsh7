"""Binary logistic regression for HW07. Author: course staff."""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
import math

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def add_ones(X: FloatArray) -> FloatArray:
    """Return a new (n, p+1) array with a leading column of ones; do not mutate X."""
    return np.column_stack((np.ones(X.shape[0]),X))

def logistic(x: FloatArray, coef: FloatArray) -> float:
    """Return sigmoid(x @ coef). x already includes its leading bias; avoid overflow."""
    wx = np.dot(x,coef)
    z = 1  / (1 + math.e ** -(wx))
    return z

def costJ(y: IntArray, y_pred: FloatArray) -> float:
    """SUM of binary negative log likelihoods (natural log), clipping p to [1e-15, 1-1e-15]."""
    cost = 0
    for i in range(len(y)):
        yi = y[i]
        pi = y_pred[i]
        clip = 1e-15
        p_clipped = np.clip(pi, clip, 1-clip)
        cost += -((yi * math.log(p_clipped)) + ((1-yi) * math.log(p_clipped)))
    return cost

def fit_SGD(X: FloatArray, y: IntArray, alpha: float, eps: float = 1e-4,
            tmax: int = 1000, seed: int = 0) -> FloatArray:
    """Initialize zero weights; shuffle paired examples each epoch with default_rng(seed).

    X has NO bias column. Return p+1 weights, bias first. After each epoch,
    compare summed cost with the previous epoch; stop if absolute change < eps.
    Always perform the first epoch. eps=0 disables early stopping. Do not mutate inputs.
    Raise ValueError for alpha<=0, eps<0, tmax<1 or empty training data.
    """
    if (alpha <= 0 or eps < 0 or tmax < 1):
        raise ValueError("either alpha, eps, or tmax out of range")

    x_with_bias = add_ones(X)
    m = x_with_bias.shape[1] #columns/features of X
    n = x_with_bias.shape[0] #rows, xi
    w = np.zeros(m)
    costs = [0]
    for i in range(tmax):

        for j in range(n):
            xi = x_with_bias[j]
            yi = y[j]
            derivative = (logistic(xi, w) - yi) * xi
            w = w - alpha * derivative
        
        y_pred = logistic(x_with_bias, w)
        
        c = costJ(y, y_pred)
        diff = abs((costs[-1]) - c)

        if diff < eps:
            costs.append(c)
            return w
        costs.append(c)
    return w


def prediction(X: FloatArray, coef: FloatArray) -> FloatArray:
    """Return n probabilities for raw (n,p) X; add the bias exactly once."""
    y_p = []
    X_w_bias = add_ones(X)
    for xi in X_w_bias:
        y_p.append(logistic(xi, coef))
    return y_p
    

def threshold(y_pred: FloatArray) -> IntArray:
    """Return labels 0/1; probability exactly 0.5 belongs to class 1."""
    y_labels = []
    for yi in y_pred:
        if yi >= 0.5:
            y_labels.append(1)
        else:
            y_labels.append(0)
    return y_labels


def accuracy(X: FloatArray, coef: FloatArray, y: IntArray) -> float:
    """Return fraction correct in [0,1]. Raise ValueError for an empty y."""
    correct = 0
    y_p = prediction(X, coef)
    y_pred = threshold(y_p)
    for i in range(y):
        if y[i] == null:
            raise ValueError("empty y")
        if y[i] == y_pred[i]:
            correct += 1
    return correct / len(y)


    
