#pragma once

#include "omp.h"
#include "common_types.hpp"


namespace generic_matrix_fns {

RowMatrixXf transpose(Eigen::Ref<RowMatrixXf> X);

RowMatrixXf naive_mmul(Eigen::Ref<RowMatrixXf> A, Eigen::Ref<RowMatrixXf> B, const int num_threads);

RowMatrixXf eigen_mmul(Eigen::Ref<RowMatrixXf> A, Eigen::Ref<RowMatrixXf> B, const int num_threads);
}

namespace eigen_utils {
    bool check_shape(Eigen::Ref<RowMatrixXf> A, Eigen::Ref<RowMatrixXf> B);
}