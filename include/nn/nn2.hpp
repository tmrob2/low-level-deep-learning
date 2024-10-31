#pragma once

#include "nn/common_types.hpp"
#include <memory>

namespace nn2 {

namespace operation {

enum OperationType {
    ReLU,
    LEAKYReLU,
    TANH,
    SIGMOID,
    LINEAR,
    WEIGHT_MULTIPLY,
    BIAS
};

struct ParamOperation {
    OperationType operationName;
    RowMatrixXf input; // This needs to be set
    RowMatrixXf output; // This needs to be calculated
    RowMatrixXf inputGrad; // This needs to be calculated on the backward pass
    RowMatrixXf outputGrad; // This needs to be set? maybe don't need
    RowMatrixXf param; // The param matrix is instantiated on construction
    RowMatrixXf paramGrad; // This needs to be calculated on the backward pass
    bool hasParam;
    ParamOperation(std::unique_ptr<RowMatrixXf> param_, OperationType instructions, bool has_param):
        operationName(instructions), hasParam(has_param),
        param(), input(), output(), inputGrad(), outputGrad(), paramGrad() {
        if (param_) {
            param = *std::move(param_);
        }
    }
};

void forward(ParamOperation& op, Eigen::Ref<RowMatrixXf> input);

void backward(ParamOperation& op, Eigen::Ref<RowMatrixXf> outputGrad);

}

namespace layer {

enum LayerType {
    DENSE,
};

struct Layer {
    RowMatrixXf input;
    RowMatrixXf output;
    RowMatrixXf inputGrad;
    bool first_time_call;
    LayerType instructions;
    std::vector<operation::ParamOperation> operations;
    int neurons_;
    operation::OperationType activation;
    Layer(LayerType name, int neurons, operation::OperationType activation_): 
        first_time_call(true), instructions(name), neurons_(neurons), 
        input(), output(), activation(activation_) {}
};

void forward(Layer& layer, Eigen::Ref<RowMatrixXf> input);
void backward(Layer& layer, Eigen::Ref<RowMatrixXf> outputGrad);
void setupLayer(Layer& layer, Eigen::Ref<RowMatrixXf> input);

}

namespace loss {

RowMatrixXf softmax(Eigen::Ref<RowMatrixXf> input);

enum LossType {
    MSE,
    CROSS_ENTROPY
};

struct LossFn {
    RowMatrixXf inputGrad;
    LossType lossType;
    float lossValue;
    float eps; // A tolerance value placeholder for some implementations
    RowMatrixXf placeholderMatrix1;
    LossFn(LossType lType): lossType(lType), inputGrad(), lossValue(0.0), eps(1e-9f) {}
};

void forward(LossFn& loss, Eigen::Ref<RowMatrixXf> prediction, Eigen::Ref<RowMatrixXf> target);

void backward(LossFn& loss, Eigen::Ref<RowMatrixXf> prediction, Eigen::Ref<RowMatrixXf> target);

}

class NeuralNetwork {
public:
    NeuralNetwork(std::vector<std::shared_ptr<layer::Layer>> layers, loss::LossType loss):
    loss_(loss::LossFn(loss)) {
        for (int i = 0; i < layers.size(); ++i) 
        {
            layers_.push_back(std::move(layers[i]));
        }
    }
    void forward(Eigen::Ref<RowMatrixXf> input);
    void backward(Eigen::Ref<RowMatrixXf> lossGrad);
    void trainBatch(Eigen::Ref<RowMatrixXf> input, Eigen::Ref<RowMatrixXf> target);
    RowMatrixXf predict(Eigen::Ref<RowMatrixXf> input);
    std::vector<std::shared_ptr<layer::Layer>>& getLayers() { return layers_; }
    float getLoss() { return lossValue; }
    Eigen::Ref<RowMatrixXf> getPredictions() { return predictions; }
    loss::LossFn& getLossFn() { return loss_; } 
protected:
    loss::LossFn loss_;
    RowMatrixXf predictions;
    float lossValue;
    std::vector<std::shared_ptr<layer::Layer>> layers_;

};
}