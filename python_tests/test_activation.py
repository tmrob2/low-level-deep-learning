import cppapi
import numpy as np

def sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))

def leaky_relu(x: np.ndarray):
    return np.maximum(0.2 * x, x)

def square(x: np.ndarray):
    return np.power(x, 2)

def test_sigmoid():
   A = np.random.randn(3, 3).astype(np.float32)
   A_ = cppapi.sigmoid(A)
   pyA_ = sigmoid(A)
   assert np.allclose(A_, pyA_)

def test_leaky_relu():
    A = np.random.randn(3, 3).astype(np.float32)
    A_ = cppapi.leaky_relu(A)
    pyA_ = leaky_relu(A)
    assert np.allclose(A_, pyA_)
    
def test_square():
    A = np.random.randn(3, 3).astype(np.float32)
    A_ = cppapi.square(A)
    pyA_ = square(A)
    assert np.allclose(A_, pyA_)
    
    
