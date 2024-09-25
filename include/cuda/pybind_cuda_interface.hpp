#pragma once

#include "cuda/cu_matrix_functions.h"
#include "nn/common_types.hpp"
#include "cuda/cu_matrix_functions.h"

/*
The purpose of this interface is to join the pybind11 types with the cuda types
avoiding a circular dependence on pybind11 in the CUDA matrix multiplication library
*/

namespace cuda_interface {
RowMatrixXf mmul(Eigen::Ref<RowMatrixXf> A, Eigen::Ref<RowMatrixXf> B, 
                 matrix_kernels::MMulAlg alg);
}

