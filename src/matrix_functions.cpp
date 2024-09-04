#include "matrix_functions.hpp"

namespace generic_matrix_fns {

RowMatrixXf transpose(Eigen::Ref<RowMatrixXf> X) {
    // This particular implementation returns a copy of the returned matrix
    RowMatrixXf output(X.cols(), X.rows());
    /* In this operation we are going to take the entire contiguous 
    block of memory and then insert it into the col of the new matrix
    therefore performing the operation [X^T]ij = [X]ji
    */
    for (int i = 0; i < X.rows(); ++i) {
        output.col(i) = X.row(i);
    }
    return output;
}


RowMatrixXf naive_mmul(Eigen::Ref<RowMatrixXf> A, Eigen::Ref<RowMatrixXf> B, const int num_threads) {
    // parallel implementation of matrix multiplication - O(n^3)
    RowMatrixXf C(A.rows(), B.cols());
    omp_set_num_threads(num_threads);
    // make a transpose of the matrix B
    auto B_t = transpose(B);
    int i, j, k;
    #pragma omp parallel for collapse(2) private(i, j, k) shared(A, B, C)
        for (int i = 0; i < A.rows(); ++i) {
            for (int j = 0; j < B_t.rows(); ++j) {
                C(i, j) = A.row(i).dot(B_t.row(j));
            }
        }
    return C;
}

RowMatrixXf eigen_mmul(Eigen::Ref<RowMatrixXf> A, Eigen::Ref<RowMatrixXf> B, const int num_threads){
    omp_set_num_threads(num_threads);
    return A * B;
}
} // namespace generic_matrix_fns

namespace eigen_utils
{
    bool check_shape(Eigen::Ref<RowMatrixXf> A, Eigen::Ref<RowMatrixXf> B) {
        return (A.rows() == B.rows()) && (A.cols() == B.cols());
    }   
} // namespace eigen_utils

