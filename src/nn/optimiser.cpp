#include "nn/nn.hpp"
#include "nn/optim.hpp"

namespace optim {

void SGD::step() {
    // iterate over the layers in a neural network
    for (auto layer: net_->getLayers()) {
        // for all of the parameters in that layer, update the parameter with its gradient weighted 
        // by the learning rate
        for (int i = 0; i < layer->getParams().size(); ++i) {
            // I think the error here is the broadcasting of the scalar-matrix multiplication.
            //printf("params: (%i, %i)\n", layer->getParams()[i]->rows(), layer->getParamGrads()[i]->cols());
            //printf("param grad: (%i, %i)\n", layer->getParamGrads()[i]->rows(), layer->getParamGrads()[i]->cols());
            *layer->getParams()[i] -= (lr * layer->getParamGrads()[i]->array()).matrix();
            //std::cout << "Param: \n" << *layer->getParams()[i] << std::endl;
        }
    }
}

}