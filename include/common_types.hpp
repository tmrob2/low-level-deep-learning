#pragma once

#include <pybind11/eigen.h>

using Eigen::Dynamic;
using Eigen::RowMajor;

using RowMatrixXf = Eigen::Matrix<float, Dynamic, Dynamic, RowMajor>;

enum Activation {
    SIGMOID,
    LEAKY_RELU,
    SQUARE
};

typedef RowMatrixXf (*ActivationFn)(RowMatrixXf);