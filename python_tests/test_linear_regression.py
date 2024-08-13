from manual_linear_regression import forward_linear_regression, loss_gradients
from sklearn.datasets import load_diabetes
import numpy as np
from typing import Dict
import cppapi
import random

def test_one_step_lr():
    diabetes = load_diabetes()
    targets = diabetes.target.astype(np.float32)
    data = diabetes.data.astype(np.float32)
    
    targets_ = targets.reshape(-1, 1)
    W = np.random.randn(data.shape[1], 1).astype(np.float32)
    random_intercept = random.random()
    weights: Dict[str, np.ndarray] = {}
    weights['W'] = W
    weights['B'] = np.array([[random_intercept]], dtype=np.float32)
    pyloss, forward_info = forward_linear_regression(data, targets_, weights)
    linear = cppapi.LinearRegression(data.shape[0], data.shape[1], random_intercept, W, 1)
    loss = linear._forward_lin_reg_one_step(data, targets_, cppapi.Loss.RMSE)
    rtol = 1e-3
    atol = 1e-3
    # checking that the forward pass is correct
    assert(np.isclose(loss, float(pyloss), rtol, atol))
    assert(np.allclose(forward_info['N'], linear.N, rtol, atol))
    assert(np.allclose(forward_info['P'], linear.P, rtol, atol))
    
    # checking that the backward pass is correct
    
    py_loss_gradient = loss_gradients(forward_info, weights)
    linear._gradient_backward_step(data, targets_)
    assert(np.allclose(linear.dLdW, py_loss_gradient['W'], rtol, atol))
    assert(np.isclose(py_loss_gradient['B'][0], linear.dLdB, rtol, atol))
    
    