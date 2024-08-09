#pragma once
#include <pybind11/eigen.h>
#include "common_types.hpp"

using Eigen::Ref;

RowMatrixXf sigmoid(RowMatrixXf x);
RowMatrixXf square(RowMatrixXf x);
RowMatrixXf leakyReLU(RowMatrixXf x);
