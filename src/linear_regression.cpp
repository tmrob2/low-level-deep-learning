#include "linear_regression.hpp"

float LinearRegression::forwardLinearRegression(Eigen::Ref<RowMatrixXf> X, 
                                                Eigen::Ref<RowMatrixXf> y,  
                                                Loss lfn_name) {
    // assert that the batch sizes of x and y are equal
    assert(X.rows() == y.rows()); // Loss function equivalence check
    assert(X.rows() == W.cols()); // Matrix multiplication check 

    auto lossFn = metrics::regression::getLossFn(lfn_name);

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

std::vector<int> LinearRegression::permutation(int size) {
    // Create the vector
    std::vector<int> numbers(size);
    // Fill the vector
    std::iota(numbers.begin(), numbers.end(), 0);
    // Shuffle the vector
    std::random_device rnd;
    std::mt19937 g(rnd());
    std::shuffle(numbers.begin(), numbers.end(), g);
    return numbers;
}

void LinearRegression::train(Eigen::Ref<RowMatrixXf> data, 
                                           Eigen::Ref<RowMatrixXf> targets, 
                                           int n_epochs, int batch_size, 
                                           Loss lossfn, float learning_rate) {
    
    /* 
    while some target tolerance has not been met
    1. select a batch of data -> this involves first generating a randomised vector 
        of indices and then iterating over chunks of these indices
    2. run the forward pass over the batch
    3. run the backward pass over the batch 
    4. use the gradients computed to update the weights W, b0
    */
    //*** We use a numpy object to instantiate the weight matrix with the Linear Regression
    // constructor *** randomly initialise the weight vector - will have weights in the interval of (-1, 1)
    //W = RowMatrixXf::Random(data.cols(), 1);
    // Randomly initialise the intercept
    std::random_device rnd;
    std::mt19937 gen(rnd());
    std::uniform_real_distribution<> dist(0, 1);
    B0 = dist(gen);
    // Resize the loss vector to the number of epochs
    std::vector<float> losses_(n_epochs);

    for (int i = 0; i < n_epochs; ++i) {
        // construct a random vector of indices which is a permutation of the total rows included
        // in the training dataset
        auto randomised_ind = permutation(data.rows());
        // iterating over the batch size -> this is a trade off of memory versus complexity
        float total_loss = 0.f;
        for (int j = 0; j < data.rows(); j += batch_size) {
            
            // determine the end index
            int end = std::min(j + batch_size, int(data.rows()));
            int actual_batch_size;
            if (j + batch_size > int(data.rows())) { actual_batch_size = (data.rows()) - j; } else { actual_batch_size = batch_size; }
            std::vector<int> batchInd(randomised_ind.begin() + j, randomised_ind.begin() + end);
            Batch batchData = selectRandomRows(batchInd, data, targets);
            float loss = forwardLinearRegression(batchData.data, batchData.targets, lossfn);
            total_loss += loss / float(actual_batch_size);
            gradients(batchData.data, batchData.targets);
            // update the weight vector and the intercept with the newly calculated values
            W -= (learning_rate * dLdW.array()).matrix(); // is the weight
            B0 -= learning_rate * dLdB; // is the intercept 
        }
        losses_[i] = total_loss;
        //printf("\nIteration: [%i], loss: [%.2f]", i, total_loss);
    }
    Eigen::Map<Eigen::VectorXf> eigenLoss(losses_.data(), losses.size());
    losses = eigenLoss;
}

RowMatrixXf LinearRegression::predict(Eigen::Ref<RowMatrixXf> data) {
    // linear regression is yhat = X * W + b0
    return (data * W).array() + B0;
}