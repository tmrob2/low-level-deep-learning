#pragma once

#include "nn/common_types.hpp"
#include "nn/matrix_functions.hpp"
#include "tutorial/comp_graph.hpp"
#include "nn/loss.hpp"
#include <vector>
#include <memory>
#include <cassert>
#include <iostream>
#include <pybind11/stl.h> // For handling standard containers

namespace hard_coded_nn {
    /*
    This namespace is different from the general template nn.
    It hardcodes two linear layers and uses a sigmoid activation function. It is
    purely an example.
    */ 
class NeuralNetwork {
public:
    RowMatrixXf W1; // The 'bunch of linear regressions' (linear layer) is (num_features, hidden_size)
    RowMatrixXf W2; // The second is the output linear layer - (hidden_size, 1)
    RowMatrixXf M1; // Dot product between the input data and weight matrix (batch_size, hidden_size)
    RowMatrixXf N1; // Addition of the first intercept
    RowMatrixXf O1; // Output of the first activation function (batch_size, hidden_size)
    RowMatrixXf M2; // The dot product of the output and second weight matrix (batch_size, 1)
    RowMatrixXf B1;
    RowMatrixXf P; // Addition of the second intercept (batch_size, 1)
    RowMatrixXf dLdW1;
    RowMatrixXf dLdW2;
    RowMatrixXf dLdB1;
    float dLdB2;
    float B2;
    NeuralNetwork(int batch_size, int num_features, int num_threads, int hidden_size,
                 Eigen::Ref<RowMatrixXf> W1_, Eigen::Ref<RowMatrixXf> W2_, Eigen::Ref<RowMatrixXf> B1_, float B2_);
    void train(Eigen::Ref<RowMatrixXf> data, Eigen::Ref<RowMatrixXf> y);
    void predict(Eigen::Ref<RowMatrixXf> data);
    float oneStepForwardPass(Activation fName, Loss lfName, 
                            Eigen::Ref<RowMatrixXf> data,
                            Eigen::Ref<RowMatrixXf> target);
    void oneStepBackwardPass(Activation fName, Eigen::Ref<RowMatrixXf> data, 
                             Eigen::Ref<RowMatrixXf> target);
    RowMatrixXf get_dLdP() { return dLdP; }
    RowMatrixXf get_dO1dN1() { return dO1dN1; }
    RowMatrixXf get_dLdN1() { return dLdN1; }
    RowMatrixXf get_dLdO1() { return dLdO1; }
private:
    RowMatrixXf dLdP;
    RowMatrixXf dO1dN1;
    RowMatrixXf dLdN1;
    RowMatrixXf dLdO1;
};
} // namespace hard_coded_nn

/*
This is the meat and potatoes of the code base - it will contain the general neural network
class
*/
namespace nn {

/// @brief An operation is an abstraction for things that the neural network performs. For example
/// weight multiply is an operation that the neural network can perform. That is, it is anything that
/// can be programmed into the computational graph.
///
/// An operation will have an input and an output
///
/// The key mistake I have been making with ownership is that the operation will never
/// own its own data. 
/// I am not even sure that a Layer will own it's own data. Only the data exclusively
/// used by the operation should be owned by the operation. Otherwise it should be 
/// shared by the neural network to the operation.
/// 
/// Ultimately the neural nework will call _forward and _backward and pass some data
/// that it owns into the operation. Technically in this case Operation will never need
/// to own data and it can just be shared temporoarily with the operation.
class Operation {
public:
    Operation() {}
    std::shared_ptr<RowMatrixXf> _forward(std::shared_ptr<RowMatrixXf> input);
    std::shared_ptr<RowMatrixXf> _backward(std::shared_ptr<RowMatrixXf> output_grad);
protected:
    virtual void _output() = 0; // this is something that will be implemented when we have a ParamOperation
    virtual void _inputGrad(std::shared_ptr<RowMatrixXf> outputGrad) = 0;
    std::shared_ptr<RowMatrixXf> input_;
    std::shared_ptr<RowMatrixXf> output_;  
    std::shared_ptr<RowMatrixXf> input_grad_;
};


class ParamOperation: public Operation {
/*
The ParamOperation extends on the Operation class but accepts a paarameter in its
constructor.
*/
public:
    ParamOperation(std::shared_ptr<RowMatrixXf> param): Operation(), param_(param), param_grad_(nullptr) {}
    std::shared_ptr<RowMatrixXf> _backward(std::shared_ptr<RowMatrixXf> outputGrad);
    friend class Layer;
protected:
    virtual void paramGrad(std::shared_ptr<RowMatrixXf> outputGrad) = 0;
    std::shared_ptr<RowMatrixXf> param_; // Param is the forward prediction of the layer
    std::shared_ptr<RowMatrixXf> param_grad_; // param grad is the partial derivative with repsect to the parameters of the layer
};

class WeightMultiply: public ParamOperation {
public:
    WeightMultiply(std::shared_ptr<RowMatrixXf> W):ParamOperation(W) {}
protected:
    void _output() override;
    void _inputGrad(std::shared_ptr<RowMatrixXf> outputGrad) override;
    void paramGrad(std::shared_ptr<RowMatrixXf> outputGrad) override;
};

class BiasAddition: public ParamOperation {
public:
    BiasAddition(std::shared_ptr<RowMatrixXf> B): ParamOperation(B) {
        assert(B.get()->rows() == 1);
    }
protected:
    void _output() override;
    void _inputGrad(std::shared_ptr<RowMatrixXf> outputGrad) override;
    void paramGrad(std::shared_ptr<RowMatrixXf> outputGrad) override;
};

namespace loss {

enum LossFns {
    MSE,
    RMSE
};

class Loss {
public:
    Loss() {}
    float forward(std::shared_ptr<RowMatrixXf> prediction, std::shared_ptr<RowMatrixXf> target);
    RowMatrixXf backward();
protected:
    virtual float _output() = 0;
    virtual RowMatrixXf _inputGrad() = 0;
    std::shared_ptr<RowMatrixXf> prediction_;
    std::shared_ptr<RowMatrixXf> target_;
    RowMatrixXf input_grad_;
};

// Make specific loss functions

class MeanSquaredError: public Loss {
public:
    MeanSquaredError(): Loss() {}
protected:
    float _output() override;
    RowMatrixXf _inputGrad() override;
};

} // namespace loss

namespace activation {

class Sigmoid: public Operation {
public:
    Sigmoid(): Operation(){}
protected:
    void _output() override;
    void _inputGrad(std::shared_ptr<RowMatrixXf> outputGrad) override;
};

} // namespace activation

/// @brief A layer of neurons in a neural network.
/// NN Layer class: forward and backward methods consist of sending the input successively forward
/// through a series of operations. Bookkeeping operations:
///
/// 1. Defining the correct series of operations in the _setup_layer function and initialising
///    and storing all of the parameters in these Operations
///
/// 2. Storing the correct vales in the input_ and output forward method
///
/// 3. Performing the correct assertion checking in the backward method 
///
/// 4. params and params_grads functions simply extract the parameters aand their gradients with
///    respect to loss from the ParamOperations within the layer
class Layer {
public:
    Layer(int neurons): neurons_(neurons), first(true) {};
    void _forward(std::shared_ptr<RowMatrixXf> input);
    void _backward(std::shared_ptr<RowMatrixXf> output_grad);
    std::vector<std::shared_ptr<RowMatrixXf>> getParams() { return params_; }
    std::vector<std::shared_ptr<RowMatrixXf>> getParamGrads() { return param_grads_; }
    bool getIsFirst() { return first; }
    void setIsFirst(bool isFirst) { first = isFirst; }
    friend class NeuralNetwork;
protected:
    virtual void setupLayer(std::shared_ptr<RowMatrixXf> input) = 0; 
    void _paramGrads();
    void _params(); 
    bool first;
    int neurons_;
    int param_operations = 0;
    std::vector<std::shared_ptr<RowMatrixXf>> params_;
    std::vector<std::shared_ptr<RowMatrixXf>> param_grads_;
    std::vector<std::shared_ptr<Operation>> operations_;
    std::vector<std::shared_ptr<Operation>> reversed_operations_;
    // cached operation variables
    // This gets a little tricky but I think the input will be shared with the neural network
    // class itself when we finally get to programming this
    // The nerual network will interface with the FFI and then everything will be shared from this
    std::shared_ptr<RowMatrixXf> input_; 
    std::shared_ptr<RowMatrixXf> output_;
    std::shared_ptr<RowMatrixXf> input_grad_;
};

/// @brief A fully connected layer which inherits from Layer
class Dense: public Layer {
public:
    Dense(int neurons, std::shared_ptr<Operation> activation): Layer(neurons) {
        // Do the setup of the layer
    }
protected:
    void setupLayer(std::shared_ptr<RowMatrixXf> input) override;
};

class NeuralNetwork {
public:
    NeuralNetwork(std::vector<std::shared_ptr<Layer>> layers, std::shared_ptr<loss::Loss> loss):
        layers_(std::move(layers)), loss_(std::move(loss)) {}
    float trainBatch(Eigen::Ref<RowMatrixXf> X, Eigen::Ref<RowMatrixXf> Y);
    std::vector<std::shared_ptr<Layer>>& getLayers() { return layers_; };
    std::shared_ptr<RowMatrixXf> forward(Eigen::Ref<RowMatrixXf> X);
    std::shared_ptr<loss::Loss> getLoss() { return loss_; }
protected:
    void backward(std::shared_ptr<RowMatrixXf> lossGrad);
    std::vector<std::shared_ptr<Layer>> layers_;
    std::shared_ptr<loss::Loss> loss_;
};


// 
// --------------------------------   TESTS   --------------------------------
//

/// @brief The tests will have to interface with python and therefore cannot be 
/// written as standalone tests.
///
/// Ok so the layer class as we will see in this test is very difficult to manage the memory between 
/// Python and C++.
namespace tests {
class TestLayerSingleOpWeightMult {
public:
    TestLayerSingleOpWeightMult(std::shared_ptr<nn::loss::Loss> loss, int neurons): 
        loss_(loss), neurons_(neurons) {}
    // We can only test forward at this point, we need a loss function to test backwards 
    void forward(Eigen::Ref<RowMatrixXf> X); 
    float partialTrain(Eigen::Ref<RowMatrixXf> X, Eigen::Ref<RowMatrixXf> target);
    Eigen::Ref<RowMatrixXf> getPrediction();
    Eigen::Ref<RowMatrixXf> getGrads();
private:
    int neurons_;
    std::shared_ptr<RowMatrixXf> prediction_;
    std::shared_ptr<RowMatrixXf> target_;
    std::shared_ptr<RowMatrixXf> output_grads_;
    std::shared_ptr<nn::loss::Loss> loss_;
    std::shared_ptr<RowMatrixXf> data_;
    std::vector<std::shared_ptr<Operation>> ops; 
};  

class TestLayerSingleOpBiasAdd {
public:
    TestLayerSingleOpBiasAdd(std::shared_ptr<nn::loss::Loss> loss, int neurons): 
        loss_(loss), neurons_(neurons) {}
    void forward(Eigen::Ref<RowMatrixXf> X); 
    float partialTrain(Eigen::Ref<RowMatrixXf> X, Eigen::Ref<RowMatrixXf> target);
    Eigen::Ref<RowMatrixXf> getPrediction();
    Eigen::Ref<RowMatrixXf> getGrads();
protected:
    int neurons_;
    std::shared_ptr<RowMatrixXf> prediction_;
    std::shared_ptr<RowMatrixXf> target_;
    std::shared_ptr<RowMatrixXf> output_grads_;
    std::shared_ptr<nn::loss::Loss> loss_;
    std::shared_ptr<RowMatrixXf> data_;
    std::vector<std::shared_ptr<Operation>> ops;
};
} // namespace tests

} // namespace nn