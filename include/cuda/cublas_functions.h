#include "cuda/cu_matrix_functions.h"
#include <cublas_v2.h>

namespace matrix_kernels {

int mmul(cublasHandle_t& handle_, FMatrix& A, FMatrix& B, 
    cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, 
    float alpha, float beta);

int colwiseSum(cublasHandle_t handle_, FMatrix& A, float* dSum);
}