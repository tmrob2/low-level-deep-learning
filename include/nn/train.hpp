#pragma once

#include "nn/nn2.hpp"

namespace optimiser {

enum OptimiserType {
    SGD
};

class Optimiser {
public:
    Optimiser(OptimiserType opType, float lr): optimiser(opType) {}
    void setNetwork(std::shared_ptr<nn2::NeuralNetwork> network);
    void step();
protected:
    OptimiserType optimiser;
    std::shared_ptr<nn2::NeuralNetwork> net;
    float lr_;
};


}

namespace train {

class Trainer {
public:
    Trainer(std::shared_ptr<nn2::NeuralNetwork> network, std::shared_ptr<optimiser::Optimiser> optimiser):
        network_(std::move(network)), optimiser_(std::move(optimiser)) {
            // set the network to be an attribute of the 
            optimiser_->setNetwork(network_);
        }
    void fit(Eigen::Ref<RowMatrixXf> Xtrain, Eigen::Ref<RowMatrixXf> Ytrain,
             Eigen::Ref<RowMatrixXf> Xtest, Eigen::Ref<RowMatrixXf> Ytest,
             int epochs, int evalEvery=10, int batchSize=32, bool restart=true, int verbose=2);
protected:
    std::shared_ptr<nn2::NeuralNetwork> network_;
    std::shared_ptr<optimiser::Optimiser> optimiser_;
};

}