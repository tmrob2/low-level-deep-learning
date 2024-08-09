import cppapi
import numpy as np
from typing import Callable, List
from test_activation import sigmoid, square, leaky_relu

def deriv(func: Callable[[np.ndarray], np.ndarray],
          input_: np.ndarray,
          delta: float = 0.001) -> np.ndarray:
    return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)

def test_derivative():
    A = np.random.randn(3, 3).astype(np.float32)
    A_ = cppapi.derivative(cppapi.Activation.SIGMOID, A, 0.001)
    pyA_ = deriv(sigmoid, A)
    rtol = 1e-4
    atol = 1e-4
    assert np.allclose(pyA_, A_, rtol, atol)
    
def test_derivative():
    A = np.random.randn(3, 3).astype(np.float32)
    A_ = cppapi.derivative(cppapi.Activation.SQUARE, A, 0.001)
    pyA_ = deriv(square, A)
    rtol = 1e-4
    atol = 1e-4
    assert np.allclose(pyA_, A_, rtol, atol)

def test_activation():
    assert cppapi.Activation.SIGMOID.name == "SIGMOID"
    
Array_Function = Callable[[np.ndarray], np.ndarray]
Chain = List[Array_Function]

def forward3(chain: Chain, input_range: np.ndarray) -> np.ndarray:
    f1 = chain[0]
    f2 = chain[1]
    f3 = chain[2]
    return f3(f2(f1(input_range)))
    
def chain_derivative2(chain: Chain, input_range: np.ndarray) -> np.ndarray:
    f1 = chain[0]
    f2 = chain[1]
    f1_of_x = f1(input_range)
    df1dx = deriv(f1, input_range)
    df2du = deriv(f2, f1_of_x)
    return df1dx * df2du

def test_chain():
    input_range = np.arange(-3, 3, 0.01).astype(np.float32)
    chain1 = [square, sigmoid]
    chain2 = [sigmoid, square]
    py_output = chain_derivative2(chain1, input_range=input_range)
    cpp_output = cppapi.chain_derivative(cppapi.Activation.SQUARE, cppapi.Activation.SIGMOID, input_range).squeeze()
    rtol = 1e-3
    atol = 1e-3
    assert np.allclose(py_output, cpp_output, rtol, atol)
    
def chain_derivative3(chain: Chain, input_range: np.ndarray) -> np.ndarray:
    f1 = chain[0]
    f2 = chain[1]
    f3 = chain[2]
    
    f1_of_x = f1(input_range)
    df1dx = deriv(f1, input_range)
    df2du = deriv(f2, f1_of_x)
    f2_of_x = f2(f1_of_x)
    df3du = deriv(f3, f2_of_x)
    return df1dx * df2du * df3du

def test_chain3():
    input_range = np.arange(-3, 3, 0.01).astype(np.float32)
    chain = [leaky_relu, square, sigmoid]
    pyoutput = chain_derivative3(chain, input_range=input_range)
    cpp_output = cppapi.chain_derivative3(cppapi.Activation.LEAKY_RELU, cppapi.Activation.SQUARE,
                                          cppapi.Activation.SIGMOID, input_range)
    rtol = 1e-3
    atol = 1e-3
    assert np.allclose(pyoutput, cpp_output.squeeze(), rtol, atol)
    
def test_forward3():
    input_range = np.arange(-3, 3, 0.01).astype(np.float32)
    chain = [leaky_relu, square, sigmoid]
    py_forward = forward3(chain, input_range)
    cpp_forward = cppapi.chain3(cppapi.Activation.LEAKY_RELU, cppapi.Activation.SQUARE,
                            cppapi.Activation.SIGMOID, input_range)
    rtol = 1e-3
    atol = 1e-3
    assert np.allclose(py_forward, cpp_forward.squeeze(), rtol, atol)
    
    
