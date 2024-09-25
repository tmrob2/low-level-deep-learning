#pragma once
#include "nn/common_types.hpp"
#include "nn/matrix_functions.hpp"

namespace metrics::regression {
    
float MSE(Eigen::Ref<RowMatrixXf> preds, Eigen::Ref<RowMatrixXf> gt);
float RMSE(Eigen::Ref<RowMatrixXf> preds, Eigen::Ref<RowMatrixXf> gt);

LossFn getLossFn(Loss lfn_name);

}
