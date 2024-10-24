#pragma once

#include "nn/nn.hpp"
#include "nn/optim.hpp"

namespace train {

class Trainer {
public:
    using NeuralNetwork = nn::NeuralNetwork;
    using Optimiser = optim::Optimiser;
    Trainer(NeuralNetwork network, std::shared_ptr<Optimiser> optimiser):
        network_(std::make_shared<NeuralNetwork>(network)), optimiser_(std::move(optimiser)) {
            // set the network to be an attribute of the 
            optimiser_->setNetwork(network_);
        }
    void fit(Eigen::Ref<RowMatrixXf> Xtrain, Eigen::Ref<RowMatrixXf> Ytrain,
             Eigen::Ref<RowMatrixXf> Xtest, Eigen::Ref<RowMatrixXf> Ytest,
             int epochs, int evalEvery=10, int batchSize=32, bool restart=true, int verbose=2);
protected:
    std::shared_ptr<NeuralNetwork> network_;
    std::shared_ptr<Optimiser> optimiser_;
};

}