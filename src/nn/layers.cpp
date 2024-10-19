#include "nn/nn.hpp"

namespace nn {
    
void Dense::setupLayer(std::shared_ptr<RowMatrixXf> input) {
    // Make a random weight matrix
    auto W = std::make_shared<RowMatrixXf>(RowMatrixXf::Random(input.get()->rows(), neurons_));
    auto bias = std::make_shared<RowMatrixXf>(RowMatrixXf::Random(1, neurons_));
    // Weights
    params_.push_back(W);
    // bias
    params_.push_back(bias);

    operations_ = {
        std::make_shared<WeightMultiply>(WeightMultiply(W)), 
        std::make_shared<BiasAddition>(bias)
    };
}

}