#pragma once

#include "common_types.hpp"
#include "matrix_functions.hpp"
#include "comp_graph.hpp"
#include "loss.hpp"

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
    Operation(): input(nullptr), output(nullptr), inputGrad(nullptr) {}
    ~Operation() {
        // Have to do the memory clean up here otherwise mem leak
        delete input;
        delete output;
        delete inputGrad;
    }
    Eigen::Ref<RowMatrixXf> forward(Eigen::Ref<RowMatrixXf> input);
    Eigen::Ref<RowMatrixXf> backward(Eigen::Ref<RowMatrixXf> output_grad);
protected:
    virtual RowMatrixXf output_() = 0;
    virtual RowMatrixXf input_() = 0;
    virtual RowMatrixXf inputGrad_(Eigen::Ref<RowMatrixXf> outputGrad) = 0;
    RowMatrixXf* input;
    RowMatrixXf* output;
    RowMatrixXf* inputGrad;
};


class ParamOperation: public Operation {
/*
The ParamOperation extends on the Operation class but accepts a paarameter in its
constructor.
*/
public:
    ParamOperation(Eigen::Ref<RowMatrixXf> param_): Operation(), param(param_), 
        paramGrad(nullptr) {}
    Eigen::Ref<RowMatrixXf> backward(Eigen::Ref<RowMatrixXf> outputGrad);
protected:
    virtual RowMatrixXf paramGrad_(Eigen::Ref<RowMatrixXf> outputGrad) = 0;
    Eigen::Ref<RowMatrixXf> param;
    RowMatrixXf* paramGrad;
};

// Now geting into the gritty parts - The actual specific operations needed to perform DL

class WeightMultiply: public ParamOperation {
public:
    WeightMultiply(Eigen::Ref<RowMatrixXf> W): ParamOperation(W) {}
protected:
    RowMatrixXf output_() override;
    RowMatrixXf inputGrad_(Eigen::Ref<RowMatrixXf> outputGrad) override;
    RowMatrixXf paramGrad_(Eigen::Ref<RowMatrixXf> outputGrad) override;
};

class BiasAddition: public ParamOperation {
public:
    BiasAddition(Eigen::Ref<RowMatrixXf> b): ParamOperation(b) {
        assert(b.rows() == 1);
    }
protected:
    RowMatrixXf output_() override;
    RowMatrixXf inputGrad_(Eigen::Ref<RowMatrixXf> outputGrad) override;
    RowMatrixXf paramGrad_(Eigen::Ref<RowMatrixXf> outputGrad) override;
};

namespace activation {

class Sigmoid: public Operation {
public:
    Sigmoid(): Operation() {}
protected:
    RowMatrixXf output_() override;
    RowMatrixXf inputGrad_(Eigen::Ref<RowMatrixXf> outputGrad) override;
};

} // namespace activation

} // namespace nn