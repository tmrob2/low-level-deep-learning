#include "cuda/cu_matrix_functions.h"
#include "cuda/cublas_functions.h"
#include <__clang_cuda_builtin_vars.h>    

void checkCublasStatus(cublasStatus_t status) {                                         
    if (status != CUBLAS_STATUS_SUCCESS) {                                              
        std::cerr << "cuBLAS Error" << std::endl;                                       
        exit(EXIT_FAILURE);                                                     
    }                                                                                   
}   

__global__ void hadamard(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        C[idx] = A[idx] * B[idx];
    }

}

namespace matrix_kernels {

/// @brief Assuming that the FMatrix device pointers are in column major format, this function
/// will compute the matrix multiplication of those two matrices A and B and store in C.
int mmul(cublasHandle_t& handle_, FMatrix& A, FMatrix& B, FMatrix& C,
    cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, 
    float alpha, float beta) {
    checkCublasStatus(cublasSgemm(handle_, transa, transb, m, n, k, 
        &alpha, A.getDeviceData(), m, B.getDeviceData(), k, 
        &beta, C.getDeviceData(), m));
}

/// @brief to perform a column wise sum of rows of a matrix we compute the product of a 
/// matrix and a vector of ones. Also assumes that dSum IS instantiated on the device to the
/// right amount of memory. 
int colwiseSum(cublasHandle_t handle_, FMatrix& A, float* dSum) {
    // 
    float alpha = 1.0; 
    float beta = 0.0;
    checkCublasStatus(cublasSgemv_v2(handle_, CUBLAS_OP_T, 
        A.getRows(), A.getCols(), &alpha, A.getDeviceData(), 
        A.getRows(), A.getOnes(), 1, &beta, dSum, 1));
}

}