## JHU NEURAL NETWORKS ASSIGNMENT 7
## Greyson Brothers

## NOTE: 
## Here I am only utilizing numpy for its array datastrucure 
## and math functions, such as exp and matmul. All neural network
## specific implementation is done by hand, as seen below. 

import numpy as np


class Differentiable:

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def d(self, *args, **kwargs):
        return self.backward(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def backward(self, *args, **kwargs):
        raise NotImplementedError
    
    def __repr__(self):
        return f"{self.__class__.__name__}(...)"
    

class Sigmoid(Differentiable):
    ## FUNCTION AT X
    def forward(self, x):
        return 1. / (1. + np.exp(-x))
    
    ## DERIVATIVE AT X
    def backward(self, x):
        y = self(x)
        return y * (1 - y)


class Error(Differentiable):
    ## SQUARED ERROR AT Y
    def forward(self, d, y):
        e = d - y
        return 0.5 * e**2
    
    ## DERIVATIVE AT Y
    def backward(self, d, y):
        return np.sum(d - y, keepdims=True)


class Layer(Differentiable):

    def __init__(self, dim_in, dim_out, use_bias=True):
        ## VARIABLES FOR FORWARD PASS
        if use_bias: dim_in += 1
        self.dim_in  = dim_in
        self.dim_out = dim_out
        self.use_bias = use_bias
        self.W = np.zeros((dim_out, dim_in))

        ## VARIABLES FOR BACKWARDS PASS
        self.x = np.zeros(dim_in)
        self.y = np.zeros(dim_out)
        self.delta = np.zeros((1, dim_out))
        self.dW = np.zeros((dim_out, dim_in))

    def set_weights(self, weights, bias=None):
        if self.use_bias:
            ## TREAT LAST WEIGHT COL AS BIAS
            assert bias is not None
            self.W[:,:-1] = weights
            self.W[:,-1:] = bias
        else:
            self.W[:] = weights
    
    def forward(self, x:np.ndarray):
        if self.use_bias:
            bias = np.ones((len(x), 1))
            x = np.append(x, bias, axis=-1)
        self.x = x
        self.y = x @ self.W.T
        return self.y

    def backward(self, e:np.ndarray, eta:float, A:Differentiable):
        ## COMPUTE DELTA K
        self.delta[:] = self.delta_k(A, self.y, e)
        ## COMPUTE WEIGHT UPDATES
        self.dW[:] = self.W + eta * self.x * self.delta.T
        ## COMPUTE THE ERROR SIGNAL TO PROPAGATE TO EACH INPUT
        e = self.delta @ self.W
        ## UPDATE WEIGHTS
        self.W[:] = self.dW
        ## PROPOGATE ERROR
        return e[0, :-1] if self.use_bias else e[0]
    
    @staticmethod
    def delta_k(A, x_k, e_k):
        ## SIMPLE CHAIN RULE - MOVE OUTSIDE OF Layer?
        return e_k * A.d(x_k) 

    def __repr__(self):
        return f"Layer({self.dim_in}, {self.dim_out})"


class NeuralNet(Differentiable):

    def __init__(
            self, 
            dim_in, 
            dim_out, 
            dim_layers=(32,), 
            activation=Sigmoid(), 
            use_bias=True,
            eta=1.0, 
        ):   
        ## DEFINE NETWORK PARAMETERS
        self.eta = eta
        self.A = activation
        
        ## DEFINE NETWORK LAYERS
        self.layers = []
        for i,dim in enumerate(dim_layers):
            ## INPUT LAYER
            if i == 0:
                self.layers.append(Layer(dim_in, dim, use_bias))
            ## HIDDEN LAYERS
            else:
                self.layers.append(Layer(dim_layers[i-1], dim, use_bias))
            ## OUTPUT LAYER
            if i == len(dim_layers)-1:
                self.layers.append(Layer(dim, dim_out, use_bias))
    
    def forward(self, x):
        for layer in self.layers:
            x = self.A(layer(x))
        return x

    def backward(self, e):
        for layer in self.layers[::-1]:
            e = layer.backward(e, self.eta, self.A)
        return e

    def __repr__(self):
        return f"NeuralNet({self.layers[0].dim_in},{','.join([str(l.dim_out) for l in self.layers])})"

        
if __name__ == "__main__":

    ## VALIDATE NN CODE AGAINST THE LECTURE EXAMPLE
    nn = NeuralNet(dim_in=2, dim_out=1, dim_layers=(2,), eta=1.0, use_bias=False)
    nn.layers[0].set_weights(weights=0.3)
    nn.layers[1].set_weights(weights=0.8)
    target = 0.7
    E = Error()
    x = np.array([1.0, 2.0])
    y = nn(x)
    e = E.d(target, y)
    nn.backward(e)

    assert np.all(np.round(nn.layers[0].W[:,0], 9) - 0.298270542 == 0), "Layer 0 input 1 weight mismatch"
    assert np.all(np.round(nn.layers[0].W[:,1], 9) - 0.296541084 == 0), "Layer 0 input 2 weight mismatch"
    assert np.all(np.round(nn.layers[1].W, 8) - 0.79252095 == 0), "Layer 1 weight mismatch"
    

    ## APPLY TO HW6 PROBLEM
    nn = NeuralNet(dim_in=2, dim_out=1, dim_layers=(2,), eta=0.1, use_bias=False)
    nn.layers[0].set_weights(weights=np.array([[0.8,0.1],[0.5,0.2]]))
    nn.layers[1].set_weights(weights=np.array([0.2, 0.7]))
    target = 0.95
    E = Error()
    x = np.array([1.0, 3.0])
    y = nn(x)
    e = E.d(target, y)
    nn.backward(e)

    ## APPLY TO HW7 PROBLEM
    nn = NeuralNet(dim_in=3, dim_out=2, dim_layers=(3,), eta=0.1, use_bias=False)
    nn.layers[0].set_weights(weights=np.random.random((3,3)))
    nn.layers[1].set_weights(weights=np.random.random((2,3)))
    target = 0.95
    E = Error()
    x = np.array([1.0, 3.0, 2.0])
    y = nn(x)
    e = E.d(target, y)
    nn.backward(e)
