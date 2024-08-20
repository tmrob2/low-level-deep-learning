#include "loss.hpp"

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
