import cppapi
import numpy as np

def test_thin_operator():
    operator = cppapi.OperatorTestClass()
    
    # Create a random 2D matrix
    A = np.random.randn(10, 3).astype(np.float32)
    B = operator.forward(A) # who has ownership over B? it's a shared pointer and will be owned by C++
    
def test_thin_param_operator():
    # create a thin param operation of the WeightMultiply ParamOperation
    X = np.random.randn(10, 2)
    neurons = 16
    W = np.random.randn(X.shape[1], neurons).astype(np.float32)
    param_operation = cppapi.ParamOperatorTestClass(W, neurons)
    param_operation.setup_layer()
    
def test_thin_param_operator_forward():
    # Testing the forward pass of the ParamOperation class
    X = np.random.randn(10, 2).astype(np.float32)
    neurons = 16
    W = np.random.randn(X.shape[1], neurons).astype(np.float32)
    param_operation = cppapi.ParamOperatorTestClass(W, neurons)
    param_operation.setup_layer()
    pred = param_operation.forward(X)
    atol = 1e-3
    rtol = 1e-3
    assert np.allclose(pred, X @ W, rtol, atol)
    
def test_thin_param_operator_backward():
    pass

def test_weight_multiply():
    pass

def test_bias_addition():
    pass
    