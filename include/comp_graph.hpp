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

RowMatrixXf chainDerivative2(Activation f1, Activation f2, Ref<RowMatrixXf> input);

RowMatrixXf chain2(Activation f1, Activation f2, Ref<RowMatrixXf> input);

RowMatrixXf chainDerivative3(Activation f1, Activation f2, Activation f3, Ref<RowMatrixXf> input);

RowMatrixXf chain3(Activation f1, Activation f2, Activation f3, Ref<RowMatrixXf> input);