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

/// @brief Stores input. Calls _output(). Output will be defined for a specific operation such as weight multiply
/// @param input 
/// @return a shared ptr of the output computation
std::shared_ptr<RowMatrixXf> Operation::_forward(std::shared_ptr<RowMatrixXf> input) {
    input_ = input;
    _output(); // the reason why we store input is becasue output 
                         // might need access to the input function
    return output_;
}

/// @brief Calls the input_grad() function
std::shared_ptr<RowMatrixXf> Operation::_backward(std::shared_ptr<RowMatrixXf> output_grad){
    eigen_utils::check_shape(output_, output_grad);
    _inputGrad(output_grad);
    eigen_utils::check_shape(input_, input_grad_);
    return input_grad_;
}

std::shared_ptr<RowMatrixXf> ParamOperation::_backward(std::shared_ptr<RowMatrixXf> outputGrad) {
    eigen_utils::check_shape(output_, outputGrad);
    _inputGrad(outputGrad);
    paramGrad(outputGrad);
    eigen_utils::check_shape(input_, input_grad_);
    eigen_utils::check_shape(param_, param_grad_);
    return input_grad_;
}

// Setting up some specific parameter based operations - The key whing here is that 
// weight multiply creates and stores its own operations.
void WeightMultiply::_output() {
    output_ = std::make_shared<RowMatrixXf>(*input_ * *param_);
}

void WeightMultiply::_inputGrad(std::shared_ptr<RowMatrixXf> outputGrad) {
    input_grad_ = std::make_shared<RowMatrixXf>(*outputGrad * param_.get()->transpose());
}

void WeightMultiply::paramGrad(std::shared_ptr<RowMatrixXf> outputGrad) {
    param_grad_ = std::make_shared<RowMatrixXf>(input_.get()->transpose() * *outputGrad);
}

void BiasAddition::_output() {
    // performs a + op
    output_ = std::make_shared<RowMatrixXf>(input_.get()->array() + param_.get()->array());
}

void BiasAddition::_inputGrad(std::shared_ptr<RowMatrixXf> outputGrad) {
    RowMatrixXf ones = RowMatrixXf::Ones(input_.get()->rows(), input_.get()->cols());
    input_grad_ = std::make_shared<RowMatrixXf>(ones * *outputGrad);
}

/// @brief Computes the parameter gradient of the bias (addition) operation
/// @param outputGrad 
void BiasAddition::paramGrad(std::shared_ptr<RowMatrixXf> outputGrad) {
    RowMatrixXf ones = RowMatrixXf::Ones(param_.get()->rows(), param_.get()->cols());
    // TODO finish and test
}

/// @brief Passes the input through the layer's operations
/// @param input 
void Layer::_forward(std::shared_ptr<RowMatrixXf> input) {

    input_ = input;
    for (auto op: operations_) {
        *input = *op.get()->_forward(input);
    }
    *output_ = *input; 
}

/// @brief Passes the out grad computed from the loss of the predition
/// backward though the set of operations of the layer. Also checks that the 
/// shapes are as expected
/// @param output_grad 
void Layer::_backward(std::shared_ptr<RowMatrixXf> output_grad) {
    assert(eigen_utils::check_shape(output_, output_grad));
    input_grad_ = output_grad; // cache the original input
    _paramGrads();
}

/// @brief Extracts the parameter gradients from the layer's operations
void Layer::_paramGrads() {
    // if the vector is empty append the operations parameter grads to the vector
    // otherwise edit the paramter gradient vectors in the vector index 
    //
    // Assert that the parameter operations have already been identified on construction of the Layer class
    assert(param_operations!=0);
    int param_elem = 0;
    for (auto op: operations_) {
        std::shared_ptr<ParamOperation> derivedElem = std::dynamic_pointer_cast<ParamOperation>(op);
        if (derivedElem) {
            *param_grads_[param_elem++] = *(derivedElem->param_grad_);
        } // Otherwise it is not a parameter operation and will not have the param_grad object
    }
}

/// @brief 
void Layer::_params() {
    assert(param_operations != 0);
    int param_elem = 0;
    for (auto op: operations_) {
        std::shared_ptr<ParamOperation> derivedElem = 
            std::dynamic_pointer_cast<ParamOperation>(op);
        if (derivedElem) {
            *params_[param_elem++] = *(derivedElem->param_);
        }
    }
}


} // namespace nn

// 
// --------------------------------   TESTS   --------------------------------
//
namespace nn::tests {

void TestLayerSingleOpWeightMult::forward(Eigen::Ref<RowMatrixXf> X) {
    data_ = std::make_shared<RowMatrixXf>(X);
    // Make a WeightMultiply operations
    // Make some random parameter data
    std::shared_ptr<RowMatrixXf> W = 
        std::make_shared<RowMatrixXf>(RowMatrixXf::Random(X.cols(), neurons_));
    WeightMultiply wm_op(W);
    ops.push_back(std::make_shared<WeightMultiply>(wm_op));
    // Now when I call forward I am doing the following
    // I set the shared pointer to the operation
    // I call the _output function of the operation which will call input_ * W
    // in this case X * W
    std::shared_ptr<RowMatrixXf> forward_output = ops[0]->_forward(data_);
    prediction_ = forward_output;
} 

float TestLayerSingleOpWeightMult::partialTrain(Eigen::Ref<RowMatrixXf> X, Eigen::Ref<RowMatrixXf> target) {
    // call forward to make a prediction using a single op
    target_ = std::make_shared<RowMatrixXf>(target);
    forward(X);
    // call the loss forward
    float loss = loss_.get()->forward(prediction_, target_);
    // call backward
    std::shared_ptr<RowMatrixXf> lossGrads = std::make_shared<RowMatrixXf>(loss_.get()->backward());
    output_grads_ = ops[0].get()->_backward(lossGrads);

    return loss;
}

Eigen::Ref<RowMatrixXf> TestLayerSingleOpWeightMult::getPrediction() {
    return *prediction_;
}

Eigen::Ref<RowMatrixXf> TestLayerSingleOpWeightMult::getGrads() {
    return *output_grads_;
}

// Make a Test for the bias operation
void TestLayerSingleOpBiasAdd::forward(Eigen::Ref<RowMatrixXf> X) {
    data_ = std::make_shared<RowMatrixXf>(X);
    // Make a WeightMultiply operations
    // Make some random parameter data
    std::shared_ptr<RowMatrixXf> W = 
        std::make_shared<RowMatrixXf>(RowMatrixXf::Random(X.cols(), neurons_));
    WeightMultiply wm_op(W);
    ops.push_back(std::make_shared<WeightMultiply>(wm_op));
    // Now when I call forward I am doing the following
    // I set the shared pointer to the operation
    // I call the _output function of the operation which will call input_ * W
    // in this case X * W
    std::shared_ptr<RowMatrixXf> forward_output = ops[0]->_forward(data_);
    // Make a WeightMultiply operations
    // Make some random parameter data
    auto B = std::make_shared<RowMatrixXf>(RowMatrixXf::Random(1, neurons_));
    BiasAddition bias(B);
    ops.push_back(std::make_shared<BiasAddition>(bias));
    // Now when I call forward I am doing the following
    // I set the shared pointer to the operation
    // I call the _output function of the operation which will call input_ * W
    // in this case X * W
    forward_output = ops[1]->_forward(forward_output);
    prediction_ = forward_output;
} 

float TestLayerSingleOpBiasAdd::partialTrain(Eigen::Ref<RowMatrixXf> X, Eigen::Ref<RowMatrixXf> target) {
    // call forward to make a prediction using a single op
    target_ = std::make_shared<RowMatrixXf>(target);
    forward(X);
    // call the loss forward
    float loss = loss_.get()->forward(prediction_, target_);
    // call backward
    std::shared_ptr<RowMatrixXf> lossGrads = std::make_shared<RowMatrixXf>(loss_.get()->backward());
    output_grads_ = ops[0].get()->_backward(lossGrads);

    return loss;
}

Eigen::Ref<RowMatrixXf> TestLayerSingleOpBiasAdd::getPrediction() {
    return *prediction_;
}

Eigen::Ref<RowMatrixXf> TestLayerSingleOpBiasAdd::getGrads() {
    return *output_grads_;
}


}