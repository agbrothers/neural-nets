## JHU NEURAL NETWORKS ASSIGNMENT 6
## Greyson Brothers
## 
## This set of problems involves using the Feed-forward, back propagation algorithm.  
## Write a basic Feedforward, multilayer neural network progam and use the values in 
## the accompanying graphic (see preceding webpage) to initialize your neural network 
## (see preceding webpage), and set the following inputs and learning parameters: 
## input x1 = 1 and input x2 = 3 with eta = 0.1, a desired output of 0.95 and the 
## weights as indicated in the graphic.  Note that since there are no biases, do not 
## include them nor update them.
##
## In the accompanying graphic of a multi-layer neural network, input x1 = 1 and 
## input x2 = 3.  Use your program to answer the following questions.

import numpy as np


def delta_k(A, x_k, e_k):
    return e_k * A.d(x_k)

def weight_update(w, eta, d_k, x_j):
    ##  W[j,k] = W[j,k] - eta*(-e_k * d_sigmoid(x_k) * x_j)
    return w + eta * d_k * x_j


class Sigmoid:
    ## FUNCTION AT X
    def __call__(self, x):
        return 1. / (1. + np.exp(-x))
    
    ## DERIVATIVE AT X
    def d(self, x):
        y = self(x)
        return y * (1 - y)


class E:
    ## SQUARED ERROR AT Y
    def __call__(self, d, y):
        e = d - y
        return 0.5 * sum(e**2)        
    
    ## DERIVATIVE AT Y
    def d(self, d, y):
        return np.sum(d - y, keepdims=True)


class Layer:

    def __init__(self, dim_in, dim_out, use_bias=True) -> None:
        ## VARIABLES FOR FORWARD PASS
        if use_bias: dim_in += 1
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.W = np.zeros((dim_out, dim_in))
        self.use_bias = use_bias

        ## VARIABLES FOR BACKWARDS PASS
        self.x = np.zeros(dim_in)
        self.y = np.zeros(dim_out)
        self.d = np.zeros(dim_out)
        self.dW = np.zeros_like(self.W)
        # self.db = np.zeros_like(self.b)

    def set_weights(self, weights, bias=None):
        if self.use_bias:
            self.W[:,:-1] = weights
            self.W[:,-1:] = bias
        else:
            self.W[:] = weights
    
    def forward(self, x:np.ndarray):
        if self.use_bias:
            x = np.append(x, 1)
        self.x = x
        self.y = self.W @ x
        return self.y

    def backward(self, lr:float, A, e:np.ndarray):
        W = self.W
        ## COMPUTE DELTA K AND UPDATED WEIGHTS
        for k,row in enumerate(W):
            ## COMPUTE DELTA K
            self.d[k] = delta_k(A, self.y[k], e[k])
            ## COMPUTE WEIGHT UPDATES
            for j,w in enumerate(W[k]):
                self.dW[k,j] = weight_update(w, lr, self.x[j], self.d[k])

        ## COMPUTE ERROR SIGNAL TO PROPAGATE
        e = np.zeros(self.dim_in)
        for j in range(self.dim_in):
            for k in range(self.dim_out):
                e[j] += self.d[k] * self.W[k,j]
        ## UPDATE WEIGHTS
        self.W[:] = self.dW
        return e

    def __call__(self, x:np.ndarray):
        return self.forward(x)
    
    def __repr__(self):
        return f"Layer({self.dim_in}, {self.dim_out})"


class NeuralNet:

    def __init__(self, dim_in, dim_out, layer_sizes=(2,), lr=1.0, activation=Sigmoid(), error=E(), use_bias=True) -> None:
        
        ## DEFINE NETWORK PARAMETERS
        self.lr = lr
        self.E = error
        self.A = activation
        self.y = None
        
        ## DEFINE NETWORK LAYERS
        self.layers = []
        for i,d in enumerate(layer_sizes):
            ## INPUT LAYER
            if i == 0:
                self.layers.append(Layer(dim_in=dim_in, dim_out=d, use_bias=use_bias))
            ## HIDDEN LAYERS
            else:
                self.layers.append(Layer(dim_in=layer_sizes[i-1], dim_out=d, use_bias=use_bias))
            ## OUTPUT LAYER
            if i == len(layer_sizes)-1:
                self.layers.append(Layer(dim_in=d, dim_out=dim_out, use_bias=use_bias))
    
    def forward(self, x):
        for layer in self.layers:
            x = self.A(layer(x))
        self.y = x
        return x

    def backward(self, d):
        e = self.E.d(d, y)
        for layer in self.layers[::-1]:
            e = layer.backward(self.lr, self.A, e)

    def __call__(self, x):
        return self.forward(x)


        
if __name__ == "__main__":

    ## VALIDATE NN CODE AGAINST THE LECTURE EXAMPLE
    nn = NeuralNet(dim_in=2, dim_out=1, layer_sizes=(2,), lr=1.0)
    nn.layers[0].set_weights(weights=0.3, bias=0.3)
    nn.layers[1].set_weights(weights=0.8, bias=0.0)
    target = 0.7
    x = np.array([1.0, 1.0])
    y = nn(x)
    nn.backward(target)

    assert np.all(np.round(nn.layers[0].W[:,:-1], 8) - 0.29827054 == 0)
    assert np.all(np.round(nn.layers[1].W[:,:-1], 8) - 0.79252095 == 0)

    ## APPLY TO HW6 PROBLEM
    nn = NeuralNet(dim_in=2, dim_out=1, layer_sizes=(2,), lr=0.1, use_bias=False)
    nn.layers[0].set_weights(weights=np.array([[0.8,0.1],[0.5,0.2]]))
    nn.layers[1].set_weights(weights=np.array([0.2, 0.7]))
    target = 0.95
    x = np.array([1.0, 3.0])
    y = nn(x)
    nn.backward(target)

    ## VALUES FOR THE QUIZ WERE OBTAINED VIA USING THE 
    ## DEBUGGER WITH BREAKPOINTS IN THE ACTIVATION AND
    ## BACKWARDS FUNCTIONS 
