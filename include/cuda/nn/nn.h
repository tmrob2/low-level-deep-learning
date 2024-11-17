#pragma once

#include <memory>
#include <vector>
#include "cuda/cu_matrix_functions.h"
#include "nn/nn2.hpp"
#include "cublas_v2.h"

namespace cuda::nn {

namespace operation {

/// The parameter operation on the device will work a bit differently to the parameter
/// operation on the CPU. First, the matrix_kernels::FMatrix are classes and therefore they
/// will all have to be constructed, this is done during the setup layer operation  
struct ParamOperation {
    nn2::operation::OperationType operationName;
    matrix_kernels::FMatrix* input;
    matrix_kernels::FMatrix* output;
    matrix_kernels::FMatrix* inputGrad;
    matrix_kernels::FMatrix* outputGrad;
    matrix_kernels::FMatrix* param;
    matrix_kernels::FMatrix* paramGrad;
    bool hasParam;
    ParamOperation() {}
};

}

namespace layer {

struct Layer {
    bool first_time_call;
    nn2::layer::LayerType instructions;
    matrix_kernels::FMatrix input;
    matrix_kernels::FMatrix output;
    matrix_kernels::FMatrix inputGrad;
    //std::vector<>
};

}

namespace loss {

}

class NeuralNetwork {
public:
    NeuralNetwork(std::vector<std::shared_ptr<layer::Layer>> layers) {}
private:
    std::vector<std::shared_ptr<layer::Layer>> layers_;
    cublasHandle_t handle_;
};

}
