#include "nn/train.hpp"
#include "data/batching.hpp"
#include "nn/nn2.hpp"

namespace train {

/// @brief Fits the neural network on the training data for a certain number of epochs. Every 
/// "eval_every" epochs, it evaluates the neural network  on the testing data.
/// @param Xtrain Complete training data
/// @param Ytrain Complete supervision data
/// @param Xtest Complete testing holdout data
/// @param Ytest Complete testing supervision data
/// @param epochs The number of epochs to train for
/// @param evalEvery The step size to evaluate at
/// @param batchSize The size to batch the complete data by
/// @param restart whether or not to restart the training of the network, 
/// i.e. reinitialising the network. 
void Trainer::fit(Eigen::Ref<RowMatrixXf> Xtrain, Eigen::Ref<RowMatrixXf> Ytrain,
                  Eigen::Ref<RowMatrixXf> Xtest, Eigen::Ref<RowMatrixXf> Ytest,
                  int epochs, int evalEvery, int batchSize, bool restart, int verbose) {
    std::shared_ptr<RowMatrixXf> Ytest_ = std::make_shared<RowMatrixXf>(Ytest);
    // This is so we can continue training the network
    if (restart) {
        for (auto layer: network_->getLayers()) {
            layer->first_time_call = true;
        }
    }

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // create some batches of data
        std::vector<std::vector<int>> batchGenerator = 
            data::createBatches(Xtrain.rows(), batchSize);
        for (int ii = 0; ii < batchGenerator.size(); ++ii) {
            // train on batches
            RowMatrixXf xTrainBatch = data::batchData(Xtrain, batchGenerator[ii]);
            RowMatrixXf yTrainBatch = data::batchData(Ytrain, batchGenerator[ii]);
            network_->trainBatch(xTrainBatch, yTrainBatch);
            if (verbose == 2) {
                printf("Epoch: %i => Loss: %f\n", epoch, network_->getLoss());
            }
            optimiser_->step();
        }
        /*if ((epoch + 1) % evalEvery == 0) {
            // do some validation
            network_->forward(Xtest);
            nn2::loss::forward(network_->getLossFn(), 
                network_->getPredictions(), Ytest);
            if (verbose >= 1) {
                printf("Validation loss after %i epochs is %.3f\n", 
                    epoch + 1, network_->getLossFn().lossValue);
            }
        }*/
    }
}
}