## JHU NEURAL NETWORKS ASSIGNMENT 5
## Greyson Brothers
import numpy as np
import math


def sigmoid(x):
    return 1. / (1. + math.exp(-x))

def descent(x, w, b, d, eta, iterations):
    
    ## COMPUTE THE PERCEPTRON DELTA FUNCTION
    for i in range(iterations): 
        activity = w.dot(x) + b
        y = sigmoid(activity)
        delta = y * (1 - y) * (d - y)
        w = w + eta*delta*x
        b = b + eta*delta
        print(f"Iter {i}: y = {y:.5f} | w = {np.round(w,4)} | b = {b:.4f}")
    return w, b


if __name__ == "__main__":

    ## Q1
    x = np.array([0.80, 0.90])
    w = np.array([0.24, 0.88])
    d = 0.95
    b = 0.
    eta = 5.0
    descent(x, w, b, d, eta, iterations=1)

    ## Q2
    descent(x, w, b, d, eta, iterations=76)

    ## Q3
    d = 0.15
    descent(x, w, b, d, eta, iterations=31)

    ## Q4
    x = 2.0
    y = 0.3
    d = 0.4
    # compute partial derivatives
    dE_de = d - y
    de_dy = -1.
    dy_dA = y * (1 - y)
    dA_d0 = 1.
    # apply the chain rule
    dE_d0 = dE_de * de_dy * dy_dA * dA_d0
    print(dE_d0)
