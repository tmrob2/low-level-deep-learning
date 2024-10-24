#pragma once
#include <vector>
#include <random>
#include <algorithm>
#include "nn/common_types.hpp"

namespace data {

std::vector<int> randomPermutation(int n);

std::vector<std::vector<int>> createBatches(int n, int batchSize);

RowMatrixXf batchData(Eigen::Ref<RowMatrixXf> X, std::vector<int>& slice);

}