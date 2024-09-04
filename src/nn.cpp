#include "nn.hpp"
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
Eigen::Ref<RowMatrixXf> Operation::forward(Eigen::Ref<RowMatrixXf> input) {
    // Stores the input and calls output
    input = input;
    output = new RowMatrixXf(output_());
    return *output;
}

Eigen::Ref<RowMatrixXf> Operation::backward(Eigen::Ref<RowMatrixXf> outputGrad) {
    assert(eigen_utils::check_shape(*output, outputGrad));
    inputGrad = new RowMatrixXf(inputGrad_(outputGrad));
    assert(eigen_utils::check_shape(*input, *inputGrad));
    return *inputGrad;
}
// --------------------
// End Operation class

// Param Operation class
// --------------------
Eigen::Ref<RowMatrixXf> ParamOperation::backward(Eigen::Ref<RowMatrixXf> outputGrad) {
    // calls the input grad and param grad methods
    // checks the shapes are appropriate
    assert(eigen_utils::check_shape(*output, outputGrad));
    inputGrad = new RowMatrixXf(inputGrad_(outputGrad));
    paramGrad = new RowMatrixXf(paramGrad_(outputGrad));
    return *inputGrad;
}
// --------------------
// End ParamOperation class

// WeightMultiply class
// --------------------
RowMatrixXf WeightMultiply::output_() {
    // compute the matrix product of the input and the param
    return *input * param;
}

RowMatrixXf WeightMultiply::inputGrad_(Eigen::Ref<RowMatrixXf> outputGrad) {
    return outputGrad * param.transpose();
}

RowMatrixXf WeightMultiply::paramGrad_(Eigen::Ref<RowMatrixXf> outputGrad) {
    return input->transpose() * outputGrad;
}
// --------------------
// End WeightMultiply class

// BiasAddition class
// --------------------
RowMatrixXf BiasAddition::output_() {
    return *input + param;
}

RowMatrixXf BiasAddition::inputGrad_(Eigen::Ref<RowMatrixXf> outputGrad) {
    // Compute the input gradient
    return RowMatrixXf::Ones(input->rows(), input->cols()) * outputGrad;
}

RowMatrixXf BiasAddition::paramGrad_(Eigen::Ref<RowMatrixXf> outputGrad) {
    // Compute the parameter gradient
    paramGrad = new RowMatrixXf(RowMatrixXf::Ones(param.rows(), param.cols()) * outputGrad);
    return paramGrad->colwise().sum(); // TODO this operation needs to be verified
}
// --------------------
// End BiasAddition class

namespace activation {

RowMatrixXf Sigmoid::output_() {
    return 1.f / (1.f + input->array().exp());
}

RowMatrixXf Sigmoid::inputGrad_(Eigen::Ref<RowMatrixXf> outputGrad) {
    // Compute the input gradient
    RowMatrixXf sigmoidBackward = (*output).array() * (1.0f - (*output).array());
    return sigmoidBackward.array() * outputGrad.array();
}

}
    
}