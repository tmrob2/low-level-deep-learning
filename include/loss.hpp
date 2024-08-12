#pragma once
#include "common_types.hpp"
#include "matrix_functions.hpp"

namespace metrics::regression {
    
float MSE(Eigen::Ref<RowMatrixXf> preds, Eigen::Ref<RowMatrixXf> gt);
float RMSE(Eigen::Ref<RowMatrixXf> preds, Eigen::Ref<RowMatrixXf> gt);

}
