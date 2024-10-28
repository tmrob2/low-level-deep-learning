#include "nn/nn2.hpp"
#include "nn/train.hpp"
//#include <iostream>
#include <memory>

void optimiser::Optimiser::step() {
    switch (optimiser) {
        case optimiser::OptimiserType::SGD:
            for (int i = 0; i < net->getLayers().size(); ++i) {
                for (int j = 0; j < net->getLayers()[i]->operations.size(); ++j) {
                    if (net->getLayers()[i]->operations[j].hasParam) {
                        //printf("param: before update: (%i, %i)\n", 
                        //    (int)net->getLayers()[i]->operations[j].param.rows(), 
                        //    (int)net->getLayers()[i]->operations[j].param.cols());
                        //std::cout << net->getLayers()[i]->operations[j].param << std::endl;
                        //printf("param grad before update: (%i, %i)\n", 
                        //    (int)net->getLayers()[i]->operations[j].paramGrad.rows(),
                        //    (int)net->getLayers()[i]->operations[j].paramGrad.cols());
                        //std::cout << net->getLayers()[i]->operations[j].paramGrad << std::endl;
                        //printf("learning rate: %f\n", lr_);
                        net->getLayers()[i]->operations[j].param = 
                            net->getLayers()[i]->operations[j].param.array() 
                            - lr_ * net->getLayers()[i]->operations[j].paramGrad.array();
                        //printf("param after update: (%i, %i)\n",
                        //    (int)net->getLayers()[i]->operations[j].param.rows(),
                        //    (int)net->getLayers()[i]->operations[j].param.cols());
                        //std::cout << net->getLayers()[i]->operations[j].param << std::endl;
                    }
                }
            }
            break;
        default:
            break;
    }
}

void optimiser::Optimiser::setNetwork(std::shared_ptr<nn2::NeuralNetwork> network_) {
    net = network_;
}