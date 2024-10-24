#include "nn/nn.hpp"

namespace nn {

/// @brief Passes the input through the layer's operations
/// @param input 
void Layer::_forward(std::shared_ptr<RowMatrixXf> input) {
    if (first) {
        setupLayer(input);
        first = false;
    }
    input_ = input;
    for (auto op: operations_) {
        input = op->_forward(input);
        
    }
    output_ = input; 
}

/// @brief Passes the out grad computed from the loss of the predition
/// backward though the set of operations of the layer. Also checks that the 
/// shapes are as expected
/// @param output_grad 
void Layer::_backward(std::shared_ptr<RowMatrixXf> output_grad) {
    assert(eigen_utils::check_shape(output_, output_grad));
    for (int i = operations_.size() - 1; i >= 0; --i) {
        std::shared_ptr<ParamOperation> derivedElem = 
            std::dynamic_pointer_cast<ParamOperation>(operations_[i]);
        if (derivedElem) {
            output_grad = derivedElem->_backward(output_grad);
        } else {
            output_grad = operations_[i]->_backward(output_grad);
        }
    }
    input_grad_ = output_grad; // cache the original input
    _paramGrads();
}

/// @brief Extracts the parameter gradients from the layer's operations
void Layer::_paramGrads() {
    // if the vector is empty append the operations parameter grads to the vector
    // otherwise edit the paramter gradient vectors in the vector index 
    //
    // Assert that the parameter operations have already been identified on construction of the Layer class
    assert(param_operations!=0);
    int param_elem = 0;
    for (auto op: operations_) {
        std::shared_ptr<ParamOperation> derivedElem = std::dynamic_pointer_cast<ParamOperation>(op);
        if (derivedElem) {
            param_grads_[param_elem++] = derivedElem->param_grad_;
        } // Otherwise it is not a parameter operation and will not have the param_grad object
    }
}

/// @brief 
void Layer::_params() {
    assert(param_operations != 0);
    int param_elem = 0;
    for (auto op: operations_) {
        std::shared_ptr<ParamOperation> derivedElem = 
            std::dynamic_pointer_cast<ParamOperation>(op);
        if (derivedElem) {
            *params_[param_elem++] = *(derivedElem->param_);
        }
    }
}
    
void Dense::setupLayer(std::shared_ptr<RowMatrixXf> input) {
    // Make a random weight matrix
    auto W = std::make_shared<RowMatrixXf>(RowMatrixXf::Random(input.get()->cols(), neurons_));
    auto bias = std::make_shared<RowMatrixXf>(RowMatrixXf::Random(1, neurons_));
    // Weights
    params_.push_back(W);
    // bias
    params_.push_back(bias);

    param_grads_.resize(params_.size());

    operations_ = {
        std::make_shared<WeightMultiply>(WeightMultiply(W)), 
        std::make_shared<BiasAddition>(bias)
    };
}

}