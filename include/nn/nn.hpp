#pragma once

#include "nn/common_types.hpp"
#include "nn/matrix_functions.hpp"
#include "tutorial/comp_graph.hpp"
#include "nn/loss.hpp"
#include <vector>
#include <memory>
#include <cassert>

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

class Operation {
/*
The purpose of this class is to handle memory operations in an efficient manner.
In particular this means that we don't want to be copying matrices all over the place.
Because deep learning is essentially matrix operations - the more that we copy matrices
the worse the peformance will be. 

Neural network layers are a series of operations followed by a non-linear operation.
- For example, it might be the weight matrix multiplication followed by a bias addition.
- This could then be followed by an activation function (non-linear operation) such as sigmoid.

*/
public:
    Operation(): input_(nullptr), output_(nullptr), input_grad_(nullptr) {}
    std::shared_ptr<RowMatrixXf> forward_(std::shared_ptr<RowMatrixXf> input);
    Eigen::Ref<RowMatrixXf> backward(Eigen::Ref<RowMatrixXf> output_grad);
protected:
    virtual RowMatrixXf output_fn() = 0;
    //virtual RowMatrixXf input_fn() = 0;
    virtual RowMatrixXf inputGrad_(Eigen::Ref<RowMatrixXf> outputGrad) = 0;
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
    ParamOperation(Eigen::Ref<RowMatrixXf> param_): Operation(), param_(param_), param_grad_(nullptr) {}
    Eigen::Ref<RowMatrixXf> backward(Eigen::Ref<RowMatrixXf> outputGrad);
    friend class Layer;
protected:
    virtual RowMatrixXf paramGrad(Eigen::Ref<RowMatrixXf> outputGrad) = 0;
    Eigen::Ref<RowMatrixXf> param_; // Param is the forward prediction of the layer
    std::shared_ptr<RowMatrixXf> param_grad_; // param grad is the partial derivative with repsect to the parameters of the layer
};

// Now geting into the gritty parts - The actual specific operations needed to perform DL

class WeightMultiply: public ParamOperation {
public:
    WeightMultiply(Eigen::Ref<RowMatrixXf> W): ParamOperation(W) {}
protected:
    RowMatrixXf output_fn() override;
    RowMatrixXf inputGrad_(Eigen::Ref<RowMatrixXf> outputGrad) override;
    RowMatrixXf paramGrad(Eigen::Ref<RowMatrixXf> outputGrad) override;
};

class BiasAddition: public ParamOperation {
public:
    BiasAddition(Eigen::Ref<RowMatrixXf> b): ParamOperation(b) {
        assert(b.rows() == 1);
    }
protected:
    RowMatrixXf output_fn() override;
    RowMatrixXf inputGrad_(Eigen::Ref<RowMatrixXf> outputGrad) override;
    RowMatrixXf paramGrad(Eigen::Ref<RowMatrixXf> outputGrad) override;
};

namespace activation {

class Sigmoid: public Operation {
public:
    Sigmoid(): Operation() {}
protected:
    RowMatrixXf output_fn() override;
    RowMatrixXf inputGrad_(Eigen::Ref<RowMatrixXf> outputGrad) override;
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
    Layer(int neurons_): neurons(neurons_){}
    RowMatrixXf forward(Eigen::Ref<RowMatrixXf> input);
    RowMatrixXf backward(Eigen::Ref<RowMatrixXf> output_grad);
protected:
    virtual void setupLayer(std::shared_ptr<RowMatrixXf> input) = 0;
    void paramGrads();
    void params();
    int neurons;
    std::shared_ptr<RowMatrixXf> input;
    RowMatrixXf output; // Caches?
    bool first;
    // ok but if these are on the GPU??? how do I store them
    // Can I have a list of GPU references - probably
    std::vector<std::shared_ptr<RowMatrixXf>> params_; // TODO do something about the memory/copies here
    std::vector<std::shared_ptr<RowMatrixXf>> param_grads_;
    // We can store both the Param operation and the Operation in this vector with pointers
    // Use a smart pointer so that we don't have to worry about the memory clean up.
    std::vector<std::shared_ptr<Operation>> operations; // operations and reveresed ops needs to be a shared pointer??
    // this needs to be instatiated at the time of creation of the operations with push_front instead of back
    std::vector<std::shared_ptr<Operation>> reversed_operatations; 
};

/// @brief A fully connected layer which inherits from Layer
class Dense: public Layer {
public:
    Dense(int neurons, std::unique_ptr<Operation> activation_): Layer(neurons), activation(std::move(activation_)) {}
protected:
    void setupLayer(std::shared_ptr<RowMatrixXf> input_) override;
    std::shared_ptr<Operation> activation;
};

namespace losses {

/// @brief The loss class for a neural network
class Loss {
public:
    Loss() {}
    float forward(Eigen::Ref<RowMatrixXf> prediction, Eigen::Ref<RowMatrixXf> target);
    std::shared_ptr<RowMatrixXf> backward();
protected:
    virtual float output() = 0;
    virtual std::shared_ptr<RowMatrixXf> inputGrad() = 0;
    std::shared_ptr<RowMatrixXf> input_grad_;
    std::shared_ptr<RowMatrixXf> prediction_;
    std::shared_ptr<RowMatrixXf> target_;
};

} // namespace losses

/// @brief The tests will have to interface with python and therefore cannot be 
/// written as standalone tests.
///
/// Ok so the layer class as we will see in this test is very difficult to manage the memory between 
/// Python and C++.
namespace tests {
/// @brief We basically want this class to store some shared pointers - do some trivial but known
/// computation with the shared pointer and then go out of scope.
///
/// Also need to think about how pybamm uses this class so the return variables need to be pybind11 safe
class ThinOperator: public Operation {
public:
    ThinOperator(): Operation() {}
    RowMatrixXf forward(Eigen::Ref<RowMatrixXf> input_);
    RowMatrixXf output_fn() override;
    RowMatrixXf inputGrad_(Eigen::Ref<RowMatrixXf> outputGrad) override;
};

} // namespace tests

} // namespace nn