#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: bp.py

import numpy as np
from src.activation import sigmoid, sigmoid_prime

def backprop(x, y, biases, weights, cost, num_layers):
    """ function of backpropagation
        Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient of all biases and weights.

        Args:
            x, y: input image x and label y
            biases, weights (list): list of biases and weights of entire network
            cost (CrossEntropyCost): object of cost computation
            num_layers (int): number of layers of the network

        Returns:
            (nabla_b, nabla_w): tuple containing the gradient for all the biases
                and weights. nabla_b and nabla_w should be the same shape as 
                input biases and weights
    """
    # initial zero list for store gradient of biases and weights
    nabla_b = [np.zeros(b.shape) for b in biases]
    nabla_w = [np.zeros(w.shape) for w in weights]
     

    ### Implement here
    # feedforward
    # Here you need to store all the activations of all the units
    # by feedforward pass
    
    activations = [x]   # List to store the activation values
    zs = []   # List to store the matrix multiplications
    val = x
    for i in range(num_layers-1):
        val =  np.dot(weights[i] , val ) + biases[i] 
        zs.append(val)
        val = sigmoid(val)
        activations.append(val) 
    ###

    # compute the gradient of error respect to output
    # activations[-1] is the list of activations of the output layer
    delta = (cost).delta(activations[-1], y)
    

    ### Implement here
    # backward pass
    # Here you need to implement the backward pass to compute the
    # gradient for each weight and bias
    
    
    # backward pass
    
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].transpose())
    
    for l in range(2, num_layers):
        z = zs[-l]
        delta = np.dot(weights[-l+1].transpose(), delta) * sigmoid_prime(z)
        nabla_b[-l] = delta
        nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
    

    return (nabla_b, nabla_w)

