#pragma once
#include <memory>
#include "nn/nn.hpp"
#include <iostream>

namespace optim {

/// Base class for neural network optimised
class Optimiser {
public: 
    using NeuralNetwork = nn::NeuralNetwork;
    Optimiser(float learningRate): lr(learningRate), net_(nullptr) {}
    void setNetwork(std::shared_ptr<NeuralNetwork> net) { net_= net; }
    virtual void step() = 0;
protected:
    float lr;
    std::shared_ptr<NeuralNetwork> net_;
};

class SGD: public Optimiser {
public: 
    SGD(float learningRate): Optimiser(learningRate) {}
    void step() override;
};

} // namespace optim