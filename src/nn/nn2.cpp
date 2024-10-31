#include "nn/nn2.hpp"
#include "Eigen/src/Core/Matrix.h"
#include "nn/common_types.hpp"
//#include <iostream>
//#include <cstdio>
#include <iostream>
#include <memory>

namespace nn2 {


namespace operation {

void forward(ParamOperation &op, Eigen::Ref<RowMatrixXf> input) {
    op.input = input;
    switch (op.operationName) {
        case OperationType::WEIGHT_MULTIPLY:
            //printf("WM param: (%i, %i)\n", (int)op.param.rows(), (int)op.param.cols());
            op.output = input * op.param;
            break;
        case OperationType::BIAS:
        {
            // make the bias addition matrix the right size
            RowMatrixXf biasTerm = RowMatrixXf::Ones(input.rows(), 1) * op.param;
            //printf("bias Term: (%i, %i)\n", (int)biasTerm.rows(), (int)biasTerm.cols());
            op.output = input + biasTerm;
            break;
        }
        case OperationType::SIGMOID:
            op.output = 1.0 / (1.0 + (-1.0 * input.array()).exp());
            break;
        case OperationType::LINEAR:
            op.output = input;
            break;
        case OperationType::ReLU:
            op.output = input.unaryExpr([] (float z) 
                 { return std::max(0.f, z); }); 
            break;
        case OperationType::LEAKYReLU:
            op.output = input.unaryExpr([] (float z) 
                { return std::max(0.2f*z, z);});
            break;
        case OperationType::TANH:
            op.output = input.array().tanh();
            break;
        default:
            break;
    }
}


void backward(ParamOperation &op, Eigen::Ref<RowMatrixXf> outputGrad) {
    switch (op.operationName) {
        case OperationType::WEIGHT_MULTIPLY:
            //printf("backward=>operation outputGrad: (%i, %i), params: (%i, %i)\n", 
            //    (int)outputGrad.rows(), (int)outputGrad.cols(),
            //    (int)op.param.transpose().rows(), (int)op.param.transpose().cols());
            op.inputGrad = outputGrad * op.param.transpose();
            op.paramGrad = op.input.transpose() * outputGrad;
            break;
        case OperationType::BIAS:
            //printf("backward=>operation input: (%i, %i) outputGrad: (%i, %i), params: (%i, %i)\n", 
            //    (int)op.input.rows(), (int)op.input.cols(), (int)outputGrad.rows(), (int)outputGrad.cols(),
            //    (int)op.param.rows(), (int)op.param.cols());
            op.inputGrad = outputGrad;
            op.paramGrad = outputGrad.colwise().sum();
            //printf("BA param grad: (%i, %i)\n", (int)op.paramGrad.rows(), (int)op.paramGrad.cols());
            break;
        case OperationType::SIGMOID:
        {
            //printf("output: (%i, %i), outputGrad: (%i, %i)\n", 
            //    (int)op.output.rows(), (int)op.output.cols(), (int)outputGrad.rows(), (int)outputGrad.cols());
            auto sigmoidBackward = op.output.array() * (1.0 - op.output.array());
            op.inputGrad = sigmoidBackward * outputGrad.array();
            break;
        }
        case OperationType::LINEAR:
            op.inputGrad = outputGrad;
            break;
        case OperationType::ReLU:
            op.inputGrad = outputGrad.unaryExpr([] (float z) 
                 { if (z > 0) { return 1.f;} else { return 0.f; } }).array();
            break;
        case OperationType::LEAKYReLU:
            op.inputGrad = outputGrad.unaryExpr([] (float z) 
                 { if (z > 0) { return 1.f;} else { return 0.2f; } }).array(); 
            break;
        case OperationType::TANH:
            op.inputGrad = outputGrad.array() * (1.0f - op.output.array().tanh().square());
            break;
        default:
            break;
    }
}

} // namespace operation

namespace layer {

void forward(Layer &layer, Eigen::Ref<RowMatrixXf> input) {
    // call setup layer if the layer has not yet been setup
    if (layer.first_time_call) {
        //printf("setup layer\n");
        setupLayer(layer, input);
        layer.first_time_call = false;
    }

    layer.input = input;

    // need to keep updating the input
    for (int i = 0; i < layer.operations.size(); ++i) {
        if (i == 0) {
            //printf("Layer Operations forward=> (%i, %i)\n", (int)input.rows(), (int)input.cols());
            operation::forward(layer.operations[i], input);
            //printf("op output: (%i, %i)\n", 
            //    (int)layer.operations[i].output.rows(), (int)layer.operations[i].output.cols());
        } else {
            //printf("Layer Operations forward=> (%i, %i)\n", 
            //    (int)layer.operations[i-1].output.rows(), (int)layer.operations[i-1].output.cols());
            operation::forward(layer.operations[i], layer.operations[i-1].output);
            //printf("op output: (%i, %i)\n", 
            //    (int)layer.operations[i].output.rows(), (int)layer.operations[i].output.cols());
        }
    }

    layer.output = layer.operations.back().output;
}

void backward(Layer &layer, Eigen::Ref<RowMatrixXf> outputGrad) {
    // set the inputGrad at the end of backward
    int counter = 0;
    for (int i = layer.operations.size() - 1; i>=0; --i) {
        //printf("Layer %i output grad: (%i, %i)", counter, 
        //    (int)outputGrad.rows(), (int)outputGrad.cols());
        if(counter==0) {
            operation::backward(layer.operations[i], outputGrad);
        } else {
            operation::backward(layer.operations[i], layer.operations[i+1].inputGrad);
        }
        counter++;
    }
    layer.inputGrad = layer.operations[0].inputGrad;
}

void setupLayer(Layer& layer, Eigen::Ref<RowMatrixXf> input) {
    switch (layer.instructions) {
        case LayerType::DENSE: 
        {
            // add a weight multiply followed by a bias addition
            RowMatrixXf W = RowMatrixXf::Random(input.cols(), layer.neurons_);
            RowMatrixXf B = RowMatrixXf::Random(1, layer.neurons_);
            layer.operations.push_back(operation::ParamOperation(
                std::make_unique<RowMatrixXf>(W), 
                operation::OperationType::WEIGHT_MULTIPLY, true));
            layer.operations.push_back(operation::ParamOperation(
                std::make_unique<RowMatrixXf>(B),
                operation::OperationType::BIAS, true));
            layer.operations.push_back(operation::ParamOperation(
                nullptr, layer.activation, false));
            break;
        }
        default:
            break;
    }
}

} // namespace layer

namespace loss {

RowMatrixXf softmax(Eigen::Ref<RowMatrixXf> input) {
    RowMatrixXf tmp = input;
    for (int i = 0; i < tmp.rows(); ++i) {
        float maxCoef = tmp.row(i).maxCoeff();
        tmp.row(i) = (tmp.row(i).array() - maxCoef).exp() 
            / (tmp.row(i).array() - maxCoef).exp().sum();
    }
    return tmp;
}

void forward(LossFn &loss, Eigen::Ref<RowMatrixXf> prediction, Eigen::Ref<RowMatrixXf> target) {
    switch (loss.lossType) {
        case LossType::MSE:
            loss.lossValue = (prediction.array() - target.array()).square().sum() 
                / (float)prediction.rows();   
            break;
        case LossType::CROSS_ENTROPY: 
        {
            loss.placeholderMatrix1 = softmax(prediction).cwiseMin(1.0f-loss.eps).cwiseMax(loss.eps);
            loss.lossValue = (-1.0 * target.array() * loss.placeholderMatrix1.array().log() 
                - (1.0f - target.array()) * (1.0f - loss.placeholderMatrix1.array()).log())
                .sum() / (float)prediction.rows();
            break;
        }
        default:
            break;
    }
    //printf("prediction: (%i, %i), target: (%i, %i)\n", 
    //    (int)prediction.rows(), (int)prediction.cols(), (int)target.rows(), (int)target.cols());
}

void backward(LossFn &loss, Eigen::Ref<RowMatrixXf> prediction, Eigen::Ref<RowMatrixXf> target) {
    switch (loss.lossType) {
        case LossType::MSE:
            loss.inputGrad = 2.0 * (prediction - target).array() 
                / (float)prediction.rows();
            break;
        case LossType::CROSS_ENTROPY:
            loss.inputGrad = loss.placeholderMatrix1 - target;
            break;
        default:
            break;
    }
}

}

void NeuralNetwork::forward(Eigen::Ref<RowMatrixXf> input) {
    for (int i = 0; i < layers_.size(); ++i) {
        //printf("layer first time call: %i\n", layers_[i]->first_time_call);
        if (i == 0) {
            //printf("input: (%i, %i)\n", (int)input.rows(), (int)input.cols());
            layer::forward(*layers_[i], input);
        } else {
            //printf("input: (%i, %i)\n", (int)layers_[i-1]->output.rows(), (int)layers_[i-1]->output.cols());
            layer::forward(*layers_[i], layers_[i - 1]->output);    
        }
    }
    predictions = layers_.back()->output; // this is a copy
}

void NeuralNetwork::backward(Eigen::Ref<RowMatrixXf> lossGrad) {
    // work backwards through the gradients
    int counter = 0;
    for (int i = layers_.size() - 1; i >= 0; --i) {
        if (counter == 0) {
            //printf("lossGrad: (%i, %i)\n", (int)lossGrad.rows(), (int)lossGrad.cols());
            layer::backward(*layers_[i], lossGrad);
            counter++;
        } else {
            //printf("prev layer loss grad: (%i, %i)\n", 
            //    (int)layers_[i+1]->inputGrad.rows(), (int)layers_[i+1]->inputGrad.cols());
            layer::backward(*layers_[i], layers_[i+1]->inputGrad);
        }
    }
}

void NeuralNetwork::trainBatch(Eigen::Ref<RowMatrixXf> input, Eigen::Ref<RowMatrixXf> target) {
    forward(input);
    // compute loss of the forward pass
    loss::forward(loss_, predictions, target);
    //printf("loss: %.3f\n", loss_.lossValue);
    lossValue = loss_.lossValue;
    loss::backward(loss_, predictions,  target);
    //printf("loss gradient\n");
    //std::cout << loss_.inputGrad << std::endl;
    backward(loss_.inputGrad);

}

RowMatrixXf NeuralNetwork::predict(Eigen::Ref<RowMatrixXf> input) {
    forward(input);
    return predictions;
}

}