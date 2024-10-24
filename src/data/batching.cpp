#include "data/batching.hpp"

namespace data {

std::vector<int> randomPermutation(int n) {
    std::vector<int> v(n);
    for (int i = 0; i < n; ++i) {
        v[i] = i;
    }

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(v.begin(), v.end(), g);
    return v;
}

std::vector<std::vector<int>> createBatches(int n, int batchSize) {
    std::vector<int> perm = randomPermutation(n);
    std::vector<std::vector<int>> batches;
    for (size_t i = 0; i < n; i += batchSize) {
        // get the next chunk
        std::vector<int> subVec;
        for (size_t j = i; j < i + batchSize && j < n;  ++j) {
            subVec.push_back(perm[j]);
        }
        if (subVec.size() == batchSize) {
            batches.push_back(subVec);
        }
    }
    return batches;
}

RowMatrixXf batchData(Eigen::Ref<RowMatrixXf> X, std::vector<int>& slice) {
    return X(slice, Eigen::all);
}

}