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