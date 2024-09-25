import cppapi
import numpy as np

def test_thin_operator():
    operator = cppapi.OperatorTestClass()
    
    # Create a random 2D matrix
    A = np.random.randn(10, 3).astype(np.float32)
    B = operator.forward(A) # who has ownership over B? it's a shared pointer and will be owned by C++
    
def test_thin_param_operator():
    pass

def test_weight_multiply():
    pass    

def test_thin_layer():
    pass
    