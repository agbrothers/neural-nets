## JHU NEURAL NETWORKS ASSIGNMENT 7
## Greyson Brothers

## NOTE: 
## Here I am only utilizing numpy for its array datastrucure 
## and math functions, such as exp and matmul. All neural network
## specific implementation is done by hand, as seen below. 

import numpy as np

from differentiable import NeuralNet, Error


def method_1(model, loss, x, y, epochs):
    batch_size = len(y)
    ## FOR EACH EPOCH
    for i in range(epochs):         
        ## UPDATE W.R.T. EACH ITEM IN BATCH
        for j in range(batch_size): 
            y_pred = model(x[j:j+1])
            e = loss.d(y[j], y_pred)
            model.backward(e)
    error = loss(y, model(x).flatten())
    return model, error

def method_2(model, loss, x, y, epochs):
    batch_size = len(y)
    ## FOR EACH ITEM IN BATCH
    for j in range(batch_size): 
        ## PERFORM AN EPOCH
        for i in range(epochs): 
            y_pred = model(x[j:j+1])
            e = loss.d(y[j], y_pred)
            model.backward(e)
    error = loss(y, model(x).flatten())
    return model, error


if __name__ == "__main__":

    ## DATASET
    x = np.array([[ 1.0,  1.0], [-1.0, -1.0]])
    y = np.array([0.9, 0.05])

    ## MODEL 1
    nn = NeuralNet(dim_in=2, dim_out=1, dim_layers=(2,), eta=1.0, use_bias=True)
    nn.layers[0].set_weights(weights=0.3, bias=0.0)
    nn.layers[1].set_weights(weights=0.8, bias=0.0)
    model_1, error_1 = method_1(model=nn, loss=Error(), x=x, y=y, epochs=15)

    ## MODEL 2
    nn = NeuralNet(dim_in=2, dim_out=1, dim_layers=(2,), eta=1.0, use_bias=True)
    nn.layers[0].set_weights(weights=0.3, bias=0.0)
    nn.layers[1].set_weights(weights=0.8, bias=0.0)    
    model_2, error_2 = method_2(model=nn, loss=Error(), x=x, y=y, epochs=15)


    ## PROBLEM 1
    print(f"Problem 1: {model_1(x)[0,0]:4f}") # 0.6583208713508027

    ## PROBLEM 2
    print(f"Problem 2: {error_1[0]:4f}") # 0.029204400612317622

    ## PROBLEM 3
    print(f"Problem 3: {model_1(x)[1,0]:4f}") # 0.3817659623378531

    ## PROBLEM 4
    print(f"Problem 4: {error_1[1]:4f}") # 0.055034326882980884

    ## PROBLEM 5
    print(f"Problem 5: {model_2(x)[0,0]:4f}") # 0.4216576338099176

    ## PROBLEM 6
    print(f"Problem 6: {error_2[0]:4f}") # 0.11440570964616345

    ## PROBLEM 7
    print(f"Problem 7: {model_2(x)[1,0]:4f}") # 0.2838448109486975

    ## PROBLEM 8
    print(f"Problem 8: {error_2[1]:4f}") # 0.027341697803816043



