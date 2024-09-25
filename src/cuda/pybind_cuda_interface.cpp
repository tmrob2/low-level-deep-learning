#include "cuda/pybind_cuda_interface.hpp"

RowMatrixXf cuda_interface::mmul(Eigen::Ref<RowMatrixXf> A, Eigen::Ref<RowMatrixXf> B, 
                                 matrix_kernels::MMulAlg alg) {
    // convert the matrix A to an FMatrix
    matrix_kernels::FMatrix matA(A.data(), A.rows(), A.cols());
    matrix_kernels::FMatrix matB(B.data(), B.rows(), B.cols());

    // create a matrix C which contains the correct size
    float* dataC = new float[matA.getRows() * matB.getCols()];
    matrix_kernels::FMatrix matC(dataC, matA.getRows(), matB.getCols());
    matA.mmul(matB, matC, alg);
    // construct an Eigen Matrix from matC
    Eigen::Map<RowMatrixXf> eigC(matC.getData(), matC.getRows(), matC.getCols());
    return eigC;
}