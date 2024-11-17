#include "nn/common_types.hpp"
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
                        /*printf("param: before update: (%i, %i)\n", 
                            (int)net->getLayers()[i]->operations[j].param.rows(), 
                            (int)net->getLayers()[i]->operations[j].param.cols());
                        std::cout << net->getLayers()[i]->operations[j].param << std::endl;
                        printf("param grad before update: (%i, %i)\n", 
                            (int)net->getLayers()[i]->operations[j].paramGrad.rows(),
                            (int)net->getLayers()[i]->operations[j].paramGrad.cols());
                        std::cout << net->getLayers()[i]->operations[j].paramGrad << std::endl;
                        printf("learning rate: %f\n", lr_);*/
                        net->getLayers()[i]->operations[j].param = 
                            net->getLayers()[i]->operations[j].param.array() 
                            - lr_ * net->getLayers()[i]->operations[j].paramGrad.array();
                        /*printf("param after update: (%i, %i)\n",
                            (int)net->getLayers()[i]->operations[j].param.rows(),
                            (int)net->getLayers()[i]->operations[j].param.cols());
                        std::cout << net->getLayers()[i]->operations[j].param << std::endl;*/
                    }
                }
            }
            break;
        case optimiser::OptimiserType::MomentumSGD:
        {
            if (!velocities_setup) {
                // setup the velocities for the first time
                for (int i = 0; i < net->getLayers().size(); ++i) {
                    for (int j = 0; j < net->getLayers()[i]->operations.size(); ++j) {
                        if (net->getLayers()[i] -> operations[j].hasParam) {
                            Eigen::Ref<RowMatrixXf> paramMatrix = 
                                net->getLayers()[i]->operations[j].param;
                            RowMatrixXf velocity(paramMatrix.rows(), paramMatrix.cols());
                            velocity.setZero();
                            velocities.push_back(velocity);
                        }
                    }
                }
            }
            int paramCounter = 0;
            for (int i = 0; i < net->getLayers().size(); ++i) {
                for (int j = 0; j < net->getLayers()[i]->operations.size(); ++j) {
                    if (net->getLayers()[i]->operations[j].hasParam) {
                        velocities[paramCounter] = 
                            (momentum_ * velocities[paramCounter].array()); 
                        velocities[paramCounter] = velocities[paramCounter].array() + 
                            lr_ * net->getLayers()[i]->operations[j].paramGrad.array();  
                        net->getLayers()[i]->operations[j].param = 
                            net->getLayers()[i]->operations[j].param.array() 
                            - velocities[paramCounter].array();
                        paramCounter++;
                    }   
                }
            }
            break;
        }
        default:
            break;
    }
}

void optimiser::Optimiser::setNetwork(std::shared_ptr<nn2::NeuralNetwork> network_) {
    net = network_;
}