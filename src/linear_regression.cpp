#include "linear_regression.hpp"
#include <assert.h>
#include <iostream>

LossFn getLossFn(Loss lfn_name) {
    if (lfn_name == Loss::MSE) {
        return &metrics::regression::MSE;
    } else if (lfn_name == Loss::RMSE) {
        return &metrics::regression::RMSE;
    } else {
        throw std::runtime_error("Unrecognised loss function");
    }
};

float LinearRegression::forwardLinearRegression(Eigen::Ref<RowMatrixXf> X, 
                                                Eigen::Ref<RowMatrixXf> y,  
                                                Loss lfn_name) {
    // assert that the batch sizes of x and y are equal
    assert(X.rows() == y.rows()); // Loss function equivalence check
    assert(X.rows() == W.cols()); // Matrix multiplication check 

    auto lossFn = getLossFn(lfn_name);

    // compute forward pass
    if (num_threads == 1) {
        N = X * W;
    } else {
        N = generic_matrix_fns::eigen_mmul(X, W, num_threads);
    }

    P = N.array() + B0;
    assert((P.rows() == y.rows()) && (P.cols() == y.cols()));
    float loss = lossFn(P, y);
    return loss;
}

void LinearRegression::gradients(Eigen::Ref<RowMatrixXf> X, Eigen::Ref<RowMatrixXf> y) {
    /*
    Comp graph

    -> |---------|          |-------- |                             
       | v(X, W)|  -> N -> |A(N + B0)|  --> P ---> Lambda(P, Y) --> L                              
    -> |---------|          |-------- |                  ^Y
                                ^B0
    Compute dLdW and dLdB for stepwise linear regression    

    From the chain rule this is 
    del Lambda(P, Y)  del A(N, B0)    del(v(X, W))
    ---------------- . ----------- . -------------
           delP           delN           delX
    */
    // dLdP shape (num_features, 1)
    auto dLdP = -2 * (y - P);
    auto dPdN = RowMatrixXf::Ones(N.rows(), 1);
    auto dPdB = RowMatrixXf::Ones(1, 1);
    // TODO introduce multi-processing here for large vectors
    auto dLdN = dLdP.cwiseProduct(dPdN);
    // tranposing the data into (num_features, batch_size)
    auto dNdW = generic_matrix_fns::transpose(X);
    // dLdW shape (num_features, 1)
    dLdW = dNdW * dLdN;
    // dLdB should be (1,)
    dLdB = (dLdP * dPdB).sum();
}

Batch LinearRegression::selectRandomRows(std::vector<int>& ind, 
                                                           Eigen::Ref<RowMatrixXf> data, 
                                                           Eigen::Ref<RowMatrixXf> targets){
    /*
    This function selects a batch of data and copies the output to a new row major matrix.
    Technically this is a temporary matrix but in our case, it will live as long as the 
    Batch object does. 
    */
    return Batch { data(ind, Eigen::all), targets(ind, Eigen::all)};
}

void LinearRegression::train(Eigen::Ref<RowMatrixXf> data, Eigen::Ref<RowMatrixXf> targets) {
    
    /* 
    while some target tolerance has not been met
    1. select a batch of data -> this involves first generating a randomised vector 
        of indices and then iterating over chunks of these indices
    2. run the forward pass over the batch
    3. run the backward pass over the batch 
    4. use the gradients computed to update the weights W, b0
    */
}