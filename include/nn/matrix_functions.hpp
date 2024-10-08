#pragma once

#include "omp.h"
#include "nn/common_types.hpp"


namespace generic_matrix_fns {

RowMatrixXf transpose(Eigen::Ref<RowMatrixXf> X);

RowMatrixXf naive_mmul(Eigen::Ref<RowMatrixXf> A, Eigen::Ref<RowMatrixXf> B, const int num_threads);

RowMatrixXf eigen_mmul(Eigen::Ref<RowMatrixXf> A, Eigen::Ref<RowMatrixXf> B, const int num_threads);
}

namespace eigen_utils {
    bool check_shape(std::shared_ptr<RowMatrixXf> A, std::shared_ptr<RowMatrixXf> B);
}