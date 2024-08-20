import cppapi
from sklearn.datasets import load_diabetes
import numpy as np
import random
from test_activation import sigmoid
from sklearn.datasets import make_regression
from typing import Callable

def deriv(func: Callable[[np.ndarray], np.ndarray],
          input_: np.ndarray,
          delta: float = 0.001) -> np.ndarray:
    return (func(input_ + delta) - func(input_ - delta)) / (2. * delta)

def test_forward_pass():
    # Computational graph of the simple hard coded neural network we are testing
    #X-> |---------|          |-------- |                        |--------|          
    #    | v(X, W)|  -> M1 -> |A(M1 + B0)|  --> sigma(M1 + B0)-> |G(O1,W2)|  -> A(M2,B2) -> P ---> Lambda(P, Y) --> L                              
    #W-> |---------|          |-------- |                        |--------|        ^B2                      ^Y
    # here sigma represents a loss function
    diabetes = load_diabetes()
    targets = diabetes.target.astype(np.float32)
    data = diabetes.data.astype(np.float32)
    targets_ = targets.reshape(-1, 1)
    
    hidden_size = 32
    batch_size = 100
    num_features = data.shape[1]
    
    W1 = np.random.randn(data.shape[1], hidden_size).astype(np.float32)
    W2 = np.random.randn(hidden_size, 1).astype(np.float32)
    B1 = np.random.randn(1, hidden_size).astype(np.float32)
    B2 = random.random()
    
    nn = cppapi.SimpleNeuralNetwork(batch_size, num_features, 1, hidden_size, W1, W2, B1, B2)
    
    loss = nn._forward_pass_one_step(cppapi.Activation.SIGMOID, cppapi.Loss.RMSE, data, targets_)
    
    # Now at this point we can access all of the objects inside the neural network class
    # so we need to make sure that a manual one step forward pass over a simple computation graph aligns
    
    M1 = np.dot(data, W1)
    atol = 1e-3
    rtol = 1e-3
    assert np.allclose(M1, nn.M1, rtol, atol)
    
    N1 = M1 + B1
    assert np.allclose(N1, nn.N1, rtol, atol)
    
    O1 = sigmoid(N1)
    assert np.allclose(O1, nn.O1, rtol, atol)
    
    M2 = np.dot(O1, W2)
    assert np.allclose(M2, nn.M2, rtol, atol)
    
    P = M2 + B2
    assert np.allclose(P, nn.P, rtol, atol)
    
    pyloss = np.sqrt(np.mean(np.power(targets_ - P, 2)))
    assert np.isclose(loss, pyloss)
    
def test_backward_pass():
    data, target = make_regression(n_samples=1000, n_features=10, noise=2, random_state=1234)
    data = data.astype(np.float32)
    targets_ = target.reshape(-1, 1).astype(np.float32)
    
    hidden_size=32
    batch_size = 100
    num_features = data.shape[1]
    
    W1 = np.random.randn(num_features, hidden_size).astype(np.float32)
    W2 = np.random.randn(hidden_size, 1).astype(np.float32)
    B1 = np.random.randn(1, hidden_size).astype(np.float32)
    B2 = random.random()

    nn = cppapi.SimpleNeuralNetwork(batch_size, num_features, 1, hidden_size, W1, W2, B1, B2)

    nn._forward_pass_one_step(cppapi.Activation.SIGMOID, cppapi.Loss.RMSE, data.astype(np.float32), targets_)
    # Manual python forward pass operations
    M1 = np.dot(data, W1)
    N1 = M1 + B1
    O1 = sigmoid(N1)
    M2 = np.dot(O1, W2)
    P = M2 + B2
    
    rtol = 1e-3
    atol = 1e-3
    
    nn._backward_pass(cppapi.Activation.SIGMOID, data, targets_)
    
    dLdP = -(targets_ - P)
    assert np.allclose(nn.get_dLdP(), dLdP, rtol, atol)
    
    dPdM2 = np.ones_like(M2)
    dLdM2 = dLdP @ dPdM2.transpose()
    dPdB2 = np.ones([1, 1])
    dLdB2 = (dLdP * dPdB2).sum(axis=0)
    
    assert np.allclose(nn.get_dLdP(), dLdP)
    assert np.isclose(dLdB2.squeeze(), nn.dLdB2)
    
    dM2dW2 = np.transpose(O1, (1, 0))
    dLdW2 = dM2dW2 @ dLdP
    
    assert np.allclose(dLdW2, nn.dLdW2)
    
    dM2dO1 = np.transpose(W2, (1, 0))
    dLdO1 = dLdP @ dM2dO1
    assert N1.shape == nn.N1.shape
    dO1dN1 = sigmoid(N1) * (1.0 - sigmoid(N1))
    assert np.allclose(dO1dN1, nn.get_dO1dN1(), rtol, atol)
    dLdN1 = dLdO1 * dO1dN1
    assert np.allclose(dLdO1, nn.get_dLdO1(), rtol, atol)
    assert np.allclose(dLdN1, nn.get_dLdN1(), rtol, atol)
    
    dLdB1 = (dLdN1 * np.ones([1, hidden_size])).sum(axis=0)
    assert np.allclose(dLdB1, nn.dLdB1, rtol, atol)
    
    dM1dW1 = np.transpose(data, (1, 0))
    dN1dM1 = np.ones_like(M1)
    dLdM1 = dLdN1 * dN1dM1
    dM1dW1 = np.transpose(data, (1, 0)) 
    dLdW1 = np.dot(dM1dW1, dLdM1)
    assert np.allclose(dLdW1, nn.dLdW1)
   