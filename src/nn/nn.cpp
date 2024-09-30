#include "nn/nn.hpp"
#include <iostream>

namespace hard_coded_nn {

NeuralNetwork::NeuralNetwork(
    int batch_size, int num_features, int num_threads, int num_outputs,
    Eigen::Ref<RowMatrixXf> W1_, Eigen::Ref<RowMatrixXf> W2_, Eigen::Ref<RowMatrixXf> B1_, float B2_): W1(W1_),
    W2(W2_), B1(B1_), B2(B2_), M1(batch_size, num_outputs), N1(batch_size, num_outputs),
    M2(num_outputs, 1), P(batch_size, 1) {}


    float NeuralNetwork::oneStepForwardPass(Activation fName, Loss lfName, 
                                            Eigen::Ref<RowMatrixXf> data, 
                                            Eigen::Ref<RowMatrixXf> target){
        
        auto f = getActivationFn(fName);
        auto lfn = metrics::regression::getLossFn(lfName);
        
        /* 
        The next line is basically the entire forward pass but not very readable
        Broken down into steps:
        X-> |---------|          |-------- |                        |--------|          
            | v(X, W)|  -> M1 -> |A(M1 + B0)|  --> sigma(M1 + B0)-> |G(O1,W2)|  -> A(M2,B2) -> P ---> Lambda(P, Y) --> L                              
        W-> |---------|          |-------- |                        |--------|        ^B2                      ^Y
        */
        M1 = data * W1;
        Eigen::RowVectorXf _b1 = B1;
        N1 = M1.rowwise() + _b1; // This is a bit of a tricky broadcast to get right
        O1 = f(N1);
        M2 = O1 * W2;
        P = M2.array() + B2;
        return lfn(target, P);
    }
    
    void NeuralNetwork::oneStepBackwardPass(Activation fName, Eigen::Ref<RowMatrixXf> data, 
                                            Eigen::Ref<RowMatrixXf> target){
        // gradients we are interested in
        // dLdB2
        // dLdW2
        // dLdB1
        // dLdW1
        dLdP = -1.0F * (target - P).array(); // in reality we won't know what this derivative is
                                                        // because the loss fn is set at runtime
        // In this example the shape of P is (N, 1) and M2 is (N, 1) so they are actually vectory
        auto dPdM2 = RowMatrixXf::Ones(M2.rows(), M2.cols());
        // use this next step as a reduction step
        auto dPdB2 = RowMatrixXf::Ones(1, 1);
        dLdB2 = (dLdP * dPdB2).sum(); // This is one of the outputs we are interested in
        auto dM2dO1 = W2.transpose();
        auto dM2dW2 = O1.transpose();
        dLdO1 = dLdP * dM2dO1;
        // dLdW2 is the computation chain dM2dW2 * dPdM2 * dLdP
        // but! dPdM2 is just 1 therefore
        // dM2dW2 * 1 * dLdP = dM2dW2 * dLdP
        dLdW2 = dM2dW2 * dLdP; 
        //dO1dN1 = derivative(fName, N1, 0.0001f);
        dO1dN1 = sigmoid(N1).array() * (1.0f - sigmoid(N1).array());
        //RowMatrixXf dN1dM1 = RowMatrixXf::Ones(M1.rows(), M1.cols());
        //RowMatrixXf dN1dB1 = RowMatrixXf::Ones(1, 32); unecessary multiplication because broadcast
        dLdN1 = dLdO1.array() * dO1dN1.array();
        dLdB1 = dLdN1.colwise().sum();
        dLdW1 = data.transpose() * dLdN1;
    }

    void NeuralNetwork::train(Eigen::Ref<RowMatrixXf> data, Eigen::Ref<RowMatrixXf> y) {}
    void NeuralNetwork::predict(Eigen::Ref<RowMatrixXf> data) {}
}

namespace nn {

// Operation class
// --------------------
// Stores the input and calls output
std::shared_ptr<RowMatrixXf> Operation::forward_(std::shared_ptr<RowMatrixXf> input) {
    input_ = input;
    output_ = std::make_shared<RowMatrixXf>(RowMatrixXf(output_fn()));
    return output_;
}

Eigen::Ref<RowMatrixXf> Operation::backward(Eigen::Ref<RowMatrixXf> outputGrad) {
    assert(eigen_utils::check_shape(*output_, outputGrad));
    input_grad_ = std::make_shared<RowMatrixXf>(RowMatrixXf(inputGrad_(std::make_shared<RowMatrixXf>(outputGrad))));
    assert(eigen_utils::check_shape(*input_, *input_grad_));
    return *input_grad_;
}
// --------------------
// End Operation class

// Param Operation class
// --------------------
// Calls the input grad and param grad methods
// and checks the shapes are appropriate
std::shared_ptr<RowMatrixXf> ParamOperation::backward(std::shared_ptr<RowMatrixXf> outputGrad) {
    assert((output_.get()->rows() && outputGrad.get()->rows()) && (output_.get()->cols() == outputGrad.get()->cols()));
    input_grad_ = std::make_shared<RowMatrixXf>(RowMatrixXf(inputGrad_(outputGrad)));
    param_grad_ = std::make_unique<RowMatrixXf>(paramGrad(outputGrad));
    return input_grad_;
}
// --------------------
// End ParamOperation class

// WeightMultiply class
// --------------------
// Computes the matrix product of the input and the param.
RowMatrixXf WeightMultiply::output_fn() {
    // TODO this is just an Eigen implementation but we should have the facility to call  
    //  whichever device and matrix multiplication necessary. 
    //  There is a question of is the matrix already on the GPU at this point?
    //  If the matrix is not on the GPU then this is a costly runtime function to call
    //  because we are transporting data back and forth on the bus. 
    return *input_ * *param_; 
}

RowMatrixXf WeightMultiply::inputGrad_(std::shared_ptr<RowMatrixXf> outputGrad) {
    return *outputGrad * param_.get()->transpose();
}

RowMatrixXf WeightMultiply::paramGrad(std::shared_ptr<RowMatrixXf> outputGrad) {
    return input_.get()->transpose() * *outputGrad;
}
// --------------------
// End WeightMultiply class

// BiasAddition class
// --------------------
RowMatrixXf BiasAddition::output_fn() {
    return *input_ + *param_;
}

RowMatrixXf BiasAddition::inputGrad_(std::shared_ptr<RowMatrixXf> outputGrad) {
    // Compute the input gradient
    return RowMatrixXf::Ones(input_->rows(), input_->cols()) * *outputGrad;
}

RowMatrixXf BiasAddition::paramGrad(std::shared_ptr<RowMatrixXf> outputGrad) {
    // Compute the parameter gradient
    param_grad_ = std::make_unique<RowMatrixXf>(
        RowMatrixXf::Ones(param_.get()-> rows(), param_.get()->cols()) * *outputGrad);
    return param_grad_->colwise().sum(); // TODO this operation needs to be verified
}
// --------------------
// End BiasAddition class

namespace activation {

RowMatrixXf Sigmoid::output_fn() {
    return 1.f / (1.f + input_->array().exp());
}

// Compute the input gradient
RowMatrixXf Sigmoid::inputGrad_(std::shared_ptr<RowMatrixXf> outputGrad) {
    RowMatrixXf sigmoidBackward = (*output_).array() * (1.0f - (*output_).array());
    return sigmoidBackward.array() * outputGrad.get()->array();
}

}

/// @brief Passes input forward through a series of operations
/// @param input Takes a 2D matrix input
/// @return Returns a 2D matrix output
RowMatrixXf Layer::forward(Eigen::Ref<RowMatrixXf> input_) {
    std::shared_ptr<RowMatrixXf> input__ = std::make_shared<RowMatrixXf>(input_);
    if (first) {
        setupLayer(input__);
        first = false;
    }
    input = input__; // Save the original input for use later

    for (auto& operation:operations) {
        // An operation regardless of whether it is a ParamOperation will always have a forward method
        input__ = operation->forward_(input); // succesively perform the forward pass over the operations
    }
    // set the output equal to the final forward transformation of the input_
    output = input_;
    return output;
}

/// @brief Passes output_grad backward through a series of operations. Also checks the 
/// the appropriate shapes. 
/// @param output_grad Input from the forward pass
/// @return Back propagated grads. 
RowMatrixXf Layer::backward(Eigen::Ref<RowMatrixXf> output_grad) {
    // assert that the input matrices and the function input are the same shape
    // will hard fail if this is not true
    // TODO check what implications this has on the python program
    assert((output.rows() == output_grad.rows()) && (output.cols() == output_grad.cols()));

    for (auto& operation: reversed_operatations) {
        output_grad = operation->backward(output_grad);
    }
    RowMatrixXf input_grad = output_grad;
    // Extract the param_grads from the operations
    paramGrads();
    return input_grad;
}

/// @brief Extracts the param_grads_ from a layer's operations
void Layer::paramGrads() {
    param_grads_.resize(operations.size());
    for (const auto& operation: operations) {
        if (dynamic_cast<ParamOperation*>(operation.get())) {
            // This is a parameter operation and we can extract the param_grad
            param_grads_.push_back(((ParamOperation*) operation.get())->param_grad_);
        }
    }
}

/// @brief Extreacts the _params from a layer's operations
void Layer::params() {
    params_.resize(operations.size());
    for (const auto& operation: operations) {
        if (dynamic_cast<ParamOperation*>(operation.get())) {
            params_.push_back(std::make_shared<RowMatrixXf>(((ParamOperation*) operation.get())->param_));
        }
    }
}

/// @brief Does the operation of a fully connected layer
/// @param input_ 
void Dense::setupLayer(std::shared_ptr<RowMatrixXf> input_) {
    // create a new random matrix and then put the matrix into the parameters. 
    params_.push_back(std::make_unique<RowMatrixXf>(RowMatrixXf::Random(input_.get()->cols(), neurons)));
    // bias
    params_.push_back(std::make_unique<RowMatrixXf>(RowMatrixXf::Random(1, neurons)));
    operations.push_back(std::make_shared<WeightMultiply>(WeightMultiply(*params_[0])));
    operations.push_back(std::make_shared<BiasAddition>(BiasAddition(*params_[1])));
    operations.push_back(activation);
    // :) This feels like an abuse of memory - RIP system ;). TODO thoroughly check this.
}

}

namespace nn::losses {

/// @brief Computes the actual loss function
/// @param prediction The prediction from the layer
/// @param target The supervised target to compare against
/// @return returns a scalar penalty for the prediction
float Loss::forward(Eigen::Ref<RowMatrixXf> prediction, Eigen::Ref<RowMatrixXf> target) {
    assert((prediction.rows() == target.rows()) && (prediction.cols() == target.cols()));
    // Save the variables
    prediction_ = std::make_shared<RowMatrixXf>(prediction);
    target_ = std::make_shared<RowMatrixXf>(target);
    return output();
}

/// @brief Compute the gradient of the loss value with respect to the input to the loss function
/// @return Matrix of gradients
std::shared_ptr<RowMatrixXf> Loss::backward() {

    input_grad_ = inputGrad();

    assert((prediction_.get()->rows() && input_grad_.get()->rows()) && 
        (prediction_.get()->cols() == input_grad_.get()->cols()));
    
    return input_grad_;
}
}

// 
// --------------------------------   TESTS   --------------------------------
//
namespace nn::tests {

RowMatrixXf ThinOperator::output_fn() {
    RowMatrixXf m = RowMatrixXf::Random(1, 1);
    return m;
}

RowMatrixXf ThinOperator::inputGrad_(std::shared_ptr<RowMatrixXf> outputGrad) {
    RowMatrixXf m = RowMatrixXf::Random(1, 1);
    return m;
}

RowMatrixXf ThinOperator::forward(Eigen::Ref<RowMatrixXf> input_) {
    std::shared_ptr<RowMatrixXf> input__ = std::make_shared<RowMatrixXf>(input_);
    std::shared_ptr<RowMatrixXf> output = std::make_shared<RowMatrixXf>(*forward_(input__));
    return *output;
}

// The thin operator should emulate the construction of the Dense Layer
// The purpose of the ThinParamOperator is to do unit tests on each of the ParamOperator methods.
void ThinParamOperator::setupLayer() {
    WeightMultiply wm(input);
    op = std::make_shared<WeightMultiply>(wm);
}

RowMatrixXf ThinParamOperator::forward(Eigen::Ref<RowMatrixXf> X) {
    std::shared_ptr<RowMatrixXf> pred = op->forward_(std::make_shared<RowMatrixXf>(X));
    return *pred;
}

RowMatrixXf ThinParamOperator::backward(Eigen::Ref<RowMatrixXf> params) {
    op->backward(params);
}

    
}