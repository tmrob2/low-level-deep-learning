import cppapi
import numpy as np

def test_layer_ops_forward():
    # create some data
    X = np.random.randn(10, 2).astype(np.float32)
    test_layer = cppapi.TestLayer(cppapi.LossFns.MSE)
    test_layer.forward(X)
    prediction = test_layer.get_prediction()
    
def test_layer_ops_partial_train():
    X = np.random.randn(10, 2).astype(np.float32)
    neurons = 16
    target = np.random.randn(10, neurons).astype(np.float32)
    test_layer = cppapi.TestLayer(cppapi.LossFns.MSE)
    test_layer.partial_train(X, target)
    prediction = test_layer.get_prediction()
    grads = test_layer.get_gradients()
    