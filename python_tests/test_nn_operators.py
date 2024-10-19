import cppapi
import numpy as np

def test_layer_ops_weight_mult_forward():
    # create some data
    neurons = 2
    X = np.random.randn(10, 2).astype(np.float32)
    mse = cppapi.MeanSquareError()
    test_layer = cppapi.TestWeightMultOp(mse, neurons)
    test_layer.forward(X)
    prediction = test_layer.get_prediction()
    
def test_layer_ops_bias_add_forward():
    neurons = 2
    X = np.random.randn(10, 2).astype(np.float32)
    mse = cppapi.MeanSquareError()
    test_layer = cppapi.TestBiasOp(mse, neurons)
    test_layer.forward(X)
    prediction = test_layer.get_prediction()
    
    
def test_layer_ops_partial_train():
    X = np.random.randn(10, 2).astype(np.float32)
    neurons = 16
    target = np.random.randn(10, neurons).astype(np.float32)
    mse = cppapi.MeanSquareError()
    test_layer = cppapi.TestWeightMultOp(mse, neurons)
    test_layer.partial_train(X, target)
    prediction = test_layer.get_prediction()
    grads = test_layer.get_gradients()
    
def test_layer_construction():
    sigmoid = cppapi.Sigmoid()
    neurons = 16
    fc1 = cppapi.Dense(neurons, sigmoid)
    