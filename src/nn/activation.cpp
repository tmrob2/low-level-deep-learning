#include "nn/activation.hpp"
#include <algorithm>
#include "nn/nn.hpp"

RowMatrixXf sigmoid(RowMatrixXf x) {
    //RowMatrixXf ones = RowMatrixXf::Ones(x.rows(), x.cols());
    RowMatrixXf result = x.unaryExpr([] (float z) {return std::exp(-z); });
    auto arr = result.array();
    return (1.f / (arr + 1.f)).matrix();
}

RowMatrixXf square(RowMatrixXf x) {
    return x.unaryExpr([] (float z) { return z * z; });
};

RowMatrixXf leakyReLU(RowMatrixXf x){
    return x.unaryExpr([] (float z) { return std::max(0.2f * z, z); });
};

namespace nn::activation {

// Activation Fns

void Sigmoid::_output() {

}

void Sigmoid::_inputGrad(std::shared_ptr<RowMatrixXf> outputGrad) {

}

}

