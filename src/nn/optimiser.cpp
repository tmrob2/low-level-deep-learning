#include "nn/train.hpp"

void optimiser::Optimiser::step() {
    switch (optimiser) {
        case optimiser::OptimiserType::SGD:
            for (int i = 0; i < net->getLayers().size(); ++i) {
                for (int j = 0; j < net->getLayers()[i]->operations.size(); ++j) {
                    if (net->getLayers()[i]->operations[j].hasParam) {
                        net->getLayers()[i]->operations[j].param = 
                            net->getLayers()[i]->operations[j].param.array() 
                            - lr_ * net->getLayers()[i]->operations[j].paramGrad.array();
                    }
                }
            }
            break;
        default:
            break;
    }
}