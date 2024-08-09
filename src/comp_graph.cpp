#include "comp_graph.hpp"
#include <iostream>


RowMatrixXf derivative(Activation fn, Ref<RowMatrixXf> input, float delta) {
    // count the copies lol
    // 1 for cp above
    // 2 every activation function uses a unary function
    // 3 conversion to array
    // 4 converion to matrix from array :( 
    // TODO make more efficient 
    // The problem is that if the first matrix is treated as immutable then
    // you have to make copies all the way down if we also want to expose the
    // the activation functions themselves - later on we won't have this problem
    // and will only expose the enum values for the activation functions.
    auto f = getActivationFn(fn);
    return ((f(input.array() + delta) - f(input.array() - delta)).array() / (2.f * delta)).matrix();
}

ActivationFn getActivationFn(Activation choice) {
    if (choice == Activation::SIGMOID) {
        return &sigmoid;
    } else if (choice == Activation::LEAKY_RELU) {
        return &leakyReLU;
    } else if (choice == Activation::SQUARE) {
        return &square;
    } else {
        throw std::runtime_error("Unrecognised activation function");
    }
};

RowMatrixXf chainDerivative2(Activation f1, Activation f2, Ref<RowMatrixXf> input) {
    auto f1_ = getActivationFn(f1);
    auto df1dx = derivative(f1, input);
    //std::cout << df1dx.transpose() << std::endl;
    auto f1_of_x = f1_(input);
    auto df2du = derivative(f2, f1_of_x);
    //std::cout << df2du.transpose() << std::endl;
    auto backward = df1dx.cwiseProduct(df2du);
    return backward;
}

RowMatrixXf chain2(Activation f1, Activation f2, Ref<RowMatrixXf> input) {
    auto f1_ = getActivationFn(f1);
    auto f2_ = getActivationFn(f2);
    return f2_(f1_(input));
}

RowMatrixXf chainDerivative3(Activation f1, Activation f2, Activation f3, Ref<RowMatrixXf> input) {
    auto f1_ = getActivationFn(f1);
    auto f2_ = getActivationFn(f2);

    auto f1_of_x = f1_(input);
    auto df1dx = derivative(f1, input);
    auto df2du = derivative(f2, f1_of_x);
    auto f2_of_x = f2_(f1_of_x);
    auto df3du = derivative(f3, f2_of_x);
    return df1dx.cwiseProduct(df2du).cwiseProduct(df3du);
}

RowMatrixXf chain3(Activation f1, Activation f2, Activation f3, Ref<RowMatrixXf> input) {
    auto f1_ = getActivationFn(f1);
    auto f2_ = getActivationFn(f2);
    auto f3_ = getActivationFn(f3);
    return f3_(f2_(f1_(input)));
}