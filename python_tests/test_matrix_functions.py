import cppapi
import numpy as np

def test_transpose():
    A = np.random.randn(3, 3).astype(np.float32)
    B = np.random.randn(5, 7).astype(np.float32)
    C = np.random.randn(15, 2).astype(np.float32)
    D = np.random.randn(2, 1).astype(np.float32)
    A_ = cppapi.transpose(A)
    pyA_ = np.transpose(A, (1, 0))
    B_ = cppapi.transpose(B)
    pyB_= np.transpose(B, (1, 0))
    C_ = cppapi.transpose(C)
    pyC_ = np.transpose(C, (1, 0))
    D_ = cppapi.transpose(D)
    pyD_ = np.transpose(D, (1, 0))
    
    assert np.array_equal(A_, pyA_)
    assert np.array_equal(B_, pyB_)
    assert np.array_equal(C_, pyC_)
    assert np.array_equal(D_, pyD_)
    
    
def test_mmul():
    A = np.random.randint(0, 10, (3, 3)).astype(np.float32)
    B = np.random.randint(0, 10, (3, 2)).astype(np.float32)
    C_ = cppapi.naive_mmul(A, B, 2)
    pyC_ = A @ B
    rtol = 1e-3
    atol = 1e-3
    assert np.allclose(C_, pyC_, rtol, atol)