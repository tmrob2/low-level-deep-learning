#pragma once

#include "common_types.hpp"
#include "matrix_functions.hpp"
#include "loss.hpp"
#include <vector>
#include <random>
#include <algorithm>


struct Batch {
    RowMatrixXf data;
    RowMatrixXf targets;
};


class LinearRegression {
public:
    RowMatrixXf W;
    RowMatrixXf N;
    RowMatrixXf P;
    RowMatrixXf dLdW;
    float dLdB;
    float B0;
    LinearRegression(int batch_size, int num_features, float intercept, Eigen::Ref<RowMatrixXf> W_,int num_threads_): 
        W(W_), N(batch_size, 1), P(batch_size, 1), B0(intercept), 
        dLdW(num_features, 1), dLdB(0.f), num_threads(num_threads_) {}; 
    float forwardLinearRegression(Eigen::Ref<RowMatrixXf> X, Eigen::Ref<RowMatrixXf> y, Loss lfn_name);
    void gradients(Eigen::Ref<RowMatrixXf> X, Eigen::Ref<RowMatrixXf> y);
    Batch selectRandomRows(std::vector<int>& ind,Eigen::Ref<RowMatrixXf> data,Eigen::Ref<RowMatrixXf> targets);
    std::vector<int> permutation(int size);
    void train(Eigen::Ref<RowMatrixXf> data, Eigen::Ref<RowMatrixXf> targets, int n_epochs, int batch_size, Loss lossfn);
private: 
    int num_threads;
};




