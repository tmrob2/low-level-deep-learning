#pragma once

#include <pybind11/eigen.h>
#include "common_types.hpp"
#include "activation.hpp"
#include "common_types.hpp"

using Eigen::Ref;

// Most basic function is computing the derivative. This is essential to 
// backpropagation

RowMatrixXf derivative(Activation act_fn, Ref<RowMatrixXf> input, float delta=0.001);

ActivationFn getActivationFn(Activation choice);

RowMatrixXf chainDerivative2(Activation act_fn1, Activation act_fn2, Ref<RowMatrixXf> input);
