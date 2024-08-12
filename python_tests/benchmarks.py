'''
This is not a Pytest script
'''
import numpy as np
import time
import cppapi

def matrix_mul_bench():

    N = 10000
    matrix_A = np.random.randn(N, N).astype(np.float32)
    matrix_B = np.random.randn(N, N).astype(np.float32)
    mem_alloc_A = matrix_A.size * matrix_A.itemsize
    mem_alloc_B = matrix_B.size * matrix_B.itemsize
    print("Memory allocation [bytes]", mem_alloc_A / 10**9 + mem_alloc_B / 10**9)
    start_time_np = time.time()
    matrix_A @ matrix_B
    end_time_np = time.time()
    print("Numpy mmul [s]", end_time_np - start_time_np)
    
    for threads in range(10, 40):
        start_time = time.time()
        cppapi.eigen_mmul(matrix_A, matrix_B, threads)
        end_time = time.time()
        print(f"C++ API No. CPUs: {threads}, runtime: {end_time - start_time}")
    
if __name__ == "__main__":
    matrix_mul_bench()
    