#include "loss.hpp"

namespace metrics::regression {
float MSE(Eigen::Ref<RowMatrixXf> preds, Eigen::Ref<RowMatrixXf> gt) {
    return (preds - gt).array().square().mean();
}

float RMSE(Eigen::Ref<RowMatrixXf> preds, Eigen::Ref<RowMatrixXf> gt){
    return std::sqrt((preds - gt).array().square().mean());
}
}
