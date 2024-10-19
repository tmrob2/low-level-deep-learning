#include "nn/loss.hpp"
#include "nn/nn.hpp"

namespace metrics::regression {
float MSE(Eigen::Ref<RowMatrixXf> preds, Eigen::Ref<RowMatrixXf> gt) {
    return (preds - gt).array().square().mean();
}

float RMSE(Eigen::Ref<RowMatrixXf> preds, Eigen::Ref<RowMatrixXf> gt){
    return std::sqrt((preds - gt).array().square().mean());
}

LossFn getLossFn(Loss lfn_name) {
    if (lfn_name == Loss::MSE) {
        return &metrics::regression::MSE;
    } else if (lfn_name == Loss::RMSE) {
        return &metrics::regression::RMSE;
    } else {
        throw std::runtime_error("Unrecognised loss function");
    }
};
}

namespace nn::loss {

/// @brief Computes the loss of the prediction
/// @param prediction 
/// @param target 
/// @return loss (f32)
float Loss::forward(std::shared_ptr<RowMatrixXf> prediction, std::shared_ptr<RowMatrixXf> target) {
    assert(eigen_utils::check_shape(prediction, target));
    prediction_ = prediction;
    target_ = target;
    return _output();
}

/// @brief Computes the gradient of the loss with respect to the input to the loss function
/// @return gradients
RowMatrixXf Loss::backward(){
    input_grad_ = _inputGrad();
    assert((prediction_.get()->rows() == input_grad_.rows() ) && (prediction_.get()->cols() == input_grad_.cols()));
    return input_grad_;
}

/// @brief Computes the per observation squared error loss
/// @return 
float MeanSquaredError::_output() {
    return (prediction_.get()->array() - target_.get()->array()).square().sum() / (float)(prediction_.get()->rows());
}

RowMatrixXf MeanSquaredError::_inputGrad() {
    return 2.0f * (prediction_.get()->array() - target_.get()->array()) / (float)prediction_.get()->rows();
}

}
