#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <Eigen/LU>

#include "nn/activation.hpp"
#include "nn/common_types.hpp"
#include "tutorial/comp_graph.hpp"
#include "nn/matrix_functions.hpp"
#include "tutorial/linear_regression.hpp"
#include "nn/nn.hpp"
#include "nn/layers.cpp"

#include "cuda/cu_matrix_functions.h"
#include "cuda/pybind_cuda_interface.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

// N.B. this would equally work with Eigen-types that are not predefined. For example replacing
// all occurrences of "Eigen::MatrixXd" with "MatD", with the following definition:
//
//  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatD;

// ---------------
// regular C++ code
// ----------------

Eigen::MatrixXd inv(const Eigen::MatrixXd &xs)
{
  return xs.inverse();
}

double det(const Eigen::MatrixXd &xs)
{
  return xs.determinant();
}

// ----------------
// Python interface
// ----------------

namespace py = pybind11;

class PyOperation: public nn::Operation {
public:
    using nn::Operation::Operation;
    void _output() override {
        PYBIND11_OVERRIDE_PURE(
            void,
            Operation,
            _output,
        );
    }

    void _inputGrad(std::shared_ptr<RowMatrixXf> outputGrad) override {
        PYBIND11_OVERRIDE_PURE(
            void,
            Operation,
            _inputGrad,
        );
    }
};

class PyLoss: public nn::loss::Loss {
public:
    using nn::loss::Loss::Loss;
    float _output() override {
        PYBIND11_OVERRIDE_PURE(
            float,
            Loss,
            _output,
        );
    }

    RowMatrixXf _inputGrad() override {
        PYBIND11_OVERRIDE_PURE(
            RowMatrixXf,
            Loss,
            _inputGrad,
        );
    }
};

PYBIND11_MODULE(_core, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: scikit_build_example

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    m.def("inv", &inv, R"pbdoc(
        Take the inverse of a matrix

        Some other explanation about the add function.
    )pbdoc");

    m.def("det", &det, R"pbdoc(
        Take the determinate of a matrix

        Some other explanation about the subtract function.
    )pbdoc");

    m.def("sigmoid", &sigmoid, R"pbdoc(
        Applies the sigmoid activation function to a matrix in 
        row major format.
    )pbdoc");

    m.def("leaky_relu", &leakyReLU,R"pbdoc(
        Applies the leaky ReLU activation function to a matrix in 
        row major format.
    )pbdoc");

    m.def("square", &square, R"pbdoc(
        Squares each element of a matrix in row major format
    )pbdoc");

    m.def("derivative", &derivative, R"pbdoc(
        Computes the derivative of a matrix using finite differences method
    )pbdoc");

    m.def("chain_derivative", &chainDerivative2,  R"pbdoc(
        Computes the chain rule (derivative) of two functions with an input matrix
    )pbdoc");

    m.def("chain", &chain2, R"pbdoc(
        Computes a composition of two functions with an input matrix
    )pbdoc");

    m.def("chain_derivative3", &chainDerivative3,  R"pbdoc(
        Computes the chain rule (derivative) of three functions with an input matrix
    )pbdoc");

    m.def("chain3", &chain3,  R"pbdoc(
        Computes a composition of three functions with an input matrix
    )pbdoc");

    m.def("transpose", &generic_matrix_fns::transpose, R"pbdoc(
        Because this is a low level lib from scratch we perform our own matrix functions
        managing the memory correctly and also optimising algorithms for memory. This 
        transpose function iterates over the rows and inserts those rows into the cols of
        a new matrix. This is because the matrix is in Eigen::RowMajor format 
    )pbdoc");

    m.def("naive_mmul", &generic_matrix_fns::naive_mmul, R"pbdoc(
        Performs CPU parallelised matrix multiplication of an Eigen::RowMajor matrix
    )pbdoc");

    m.def("eigen_mmul", &generic_matrix_fns::eigen_mmul, R"pbdoc(
        Performs Eigen's in-build OMP parallelised matrix multiplication of an 
        Eigen::RowMajor matrix
    )pbdoc");

    m.def("multi_input_foward_sum", &multiInputForwardSum, R"pbdoc(
        This function performs the backpropagation through a simple computation
        graph taking as input multiple 2D matrices. An important step for deep learning.
    )pbdoc");
    
    // Accessible objects
    // Defing the Linear Regression object for performing stepped linear regression based on
    // a computational graph
    py::class_<LinearRegression>(m, "LinearRegression")
        .def(py::init<int, int, float, Eigen::Ref<RowMatrixXf>, int>())
        .def("_forward_lin_reg_one_step", &LinearRegression::forwardLinearRegression, R"pbdoc(
            Performs one step foward of linear regression
        )pbdoc")
        .def("_gradient_backward_step", &LinearRegression::gradients, R"pbdoc(
            Performs a backward step of computing the gradients of the output with respect to
            the weights and the intercept.
        )pbdoc")
        .def("train", &LinearRegression::train, R"pbdoc(
            trains the weights and the intercept using an iterative method and a supervised dataset
        )pbdoc")
        .def("predict", &LinearRegression::predict, R"pbdoc(
            given a trained weight matrix and intercept - predicts the target values for an input dataset
        )pbdoc")
        .def_readonly("N", &LinearRegression::N)
        .def_readonly("P", &LinearRegression::P)
        .def_readonly("W", &LinearRegression::W)
        .def_readonly("B0", &LinearRegression::B0)
        .def_readonly("dLdB", &LinearRegression::dLdB)
        .def_readonly("dLdW", &LinearRegression::dLdW);

    // Defining a simple hardcoded neural network for testing in the python interface
    py::class_<hard_coded_nn::NeuralNetwork>(m, "SimpleNeuralNetwork")
        .def(py::init<int, int, int, int, Eigen::Ref<RowMatrixXf>, Eigen::Ref<RowMatrixXf>,
             Eigen::Ref<RowMatrixXf>, float>())
        .def("_forward_pass_one_step", &hard_coded_nn::NeuralNetwork::oneStepForwardPass, R"pbdoc(
            performs the forward pass of the hardcoded computation graph
            This function is only exposed for testing
        )pbdoc")
        .def("_backward_pass", &hard_coded_nn::NeuralNetwork::oneStepBackwardPass, R"pbdoc(
            Performs one backpropagation step of the computation graph
            This function is only exposed for testing
        )pbdoc")
        .def("train", &hard_coded_nn::NeuralNetwork::train, R"pbdoc(
             trains the neural network given the dataset, loss function, and activation function
        )pbdoc")
        .def("predict", &hard_coded_nn::NeuralNetwork::predict, R"pbdoc(
             With learned weight matrices W1, W2, B1, B2 the function predicts an expected
             response given some input data
        )pbdoc")
        .def("get_dLdP", &hard_coded_nn::NeuralNetwork::get_dLdP)
        .def("get_dO1dN1", &hard_coded_nn::NeuralNetwork::get_dO1dN1)
        .def("get_dLdN1", &hard_coded_nn::NeuralNetwork::get_dLdN1)
        .def("get_dLdO1", &hard_coded_nn::NeuralNetwork::get_dLdO1)
        .def_readonly("W1", &hard_coded_nn::NeuralNetwork::W1)
        .def_readonly("W2", &hard_coded_nn::NeuralNetwork::W2)
        .def_readonly("M1", &hard_coded_nn::NeuralNetwork::M1)
        .def_readonly("M2", &hard_coded_nn::NeuralNetwork::M2)
        .def_readonly("N1", &hard_coded_nn::NeuralNetwork::N1)
        .def_readonly("O1", &hard_coded_nn::NeuralNetwork::O1)
        .def_readonly("W2", &hard_coded_nn::NeuralNetwork::W2)
        .def_readonly("P", &hard_coded_nn::NeuralNetwork::P)
        .def_readonly("B1", &hard_coded_nn::NeuralNetwork::B1)
        .def_readonly("B2", &hard_coded_nn::NeuralNetwork::B2)
        .def_readonly("dLdW1", &hard_coded_nn::NeuralNetwork::dLdW1)
        .def_readonly("dLdW2", &hard_coded_nn::NeuralNetwork::dLdW2)
        .def_readonly("dLdB1", &hard_coded_nn::NeuralNetwork::dLdB1)
        .def_readonly("dLdB2", &hard_coded_nn::NeuralNetwork::dLdB2);

    py::class_<nn::loss::Loss, PyLoss, std::shared_ptr<nn::loss::Loss>>(m, "LossFn")
        .def(py::init<>());

    py::class_<nn::loss::MeanSquaredError, nn::loss::Loss, std::shared_ptr<nn::loss::MeanSquaredError>>(m, "MeanSquareError")
        .def(py::init<>());

    py::class_<nn::tests::TestLayerSingleOpWeightMult>(m, "TestWeightMultOp")
        .def(py::init<std::shared_ptr<nn::loss::Loss>, int>())
        .def("forward", &nn::tests::TestLayerSingleOpWeightMult::forward, py::return_value_policy::reference, R"pbdoc(
            Calls the forward pass of the TestLayer. Stores some data as a shared pointer
            creates an operation (WeightMultiply) does the operations of Weight
            Multiply and then returns a reference to an Eigen matrix for numpy to use.
        )pbdoc")
        .def("partial_train", &nn::tests::TestLayerSingleOpWeightMult::partialTrain, R"pbdoc(
            Does a forward pass through a WeightMultiply operation to get the prediction from the 
            operation.
            Uses the loss function input into the test class to perform the partial derivatives
            with respect to the prediction from the forward pass
            Does Back propagation with respect to the WeightMultiply operation. 
            :returns: Returns the loss (f32) with respect to the partial derivative of the Loss function
        )pbdoc")
        .def("get_prediction", &nn::tests::TestLayerSingleOpWeightMult::getPrediction, py::return_value_policy::reference, R"pbdoc(
            Returns the prediction matrix as an Eigen matrix reference that is owned by the test
            class. Can be used directly in numpy
        )pbdoc")
        .def("get_gradients", &nn::tests::TestLayerSingleOpWeightMult::getGrads, py::return_value_policy::reference, R"pbdoc(
            Returns the gradients matrix of the partial derivatives with respect the the WeightMultiply
            operation i.e. X (training data input)  according to the loss function input 
            into the test class. 
        )pbdoc");

    py::class_<nn::tests::TestLayerSingleOpBiasAdd>(m, "TestBiasOp")
        .def(py::init<std::shared_ptr<nn::loss::Loss>, int>())
        .def("forward", &nn::tests::TestLayerSingleOpBiasAdd::forward, py::return_value_policy::reference, R"pbdoc(
            Calls the forward pass of the TestLayer. Stores some data as a shared pointer
            creates an operation (BiasOp) does the operations of Weight
            Multiply and then returns a reference to an Eigen matrix for numpy to use.
        )pbdoc")
        .def("partial_train", &nn::tests::TestLayerSingleOpBiasAdd::partialTrain, R"pbdoc(
            Does a forward pass through a Bias Add operation to get the prediction from the 
            operation.
            Uses the loss function input into the test class to perform the partial derivatives
            with respect to the prediction from the forward pass
            Does Back propagation with respect to the Bias Add operation. 
            :returns: Returns the loss (f32) with respect to the partial derivative of the Loss function
        )pbdoc")
        .def("get_prediction", &nn::tests::TestLayerSingleOpBiasAdd::getPrediction, py::return_value_policy::reference, R"pbdoc(
            Returns the prediction matrix as an Eigen matrix reference that is owned by the test
            class. Can be used directly in numpy
        )pbdoc")
        .def("get_gradients", &nn::tests::TestLayerSingleOpBiasAdd::getGrads, py::return_value_policy::reference, R"pbdoc(
            Returns the gradients matrix of the partial derivatives with respect the the Bias Op
            operation i.e. X (training data input)  according to the loss function input 
            into the test class. 
        )pbdoc");

    py::class_<nn::Operation, PyOperation, std::shared_ptr<nn::Operation>>(m, "Operation")
        .def(py::init<>());

    py::class_<nn::activation::Sigmoid, nn::Operation, std::shared_ptr<nn::activation::Sigmoid>>(m, "Sigmoid")
        .def(py::init<>());

    py::class_<nn::Dense>(m, "Dense")
        .def(py::init<int, std::shared_ptr<nn::Operation>>());

    // Exposing the Enum for selecting the Activation functions
    py::enum_<Activation>(m, "Activation")
        .value("SIGMOID", Activation::SIGMOID)
        .value("SQUARE", Activation::SQUARE)
        .value("LEAKY_RELU", Activation::LEAKY_RELU)
        .export_values();

    // Exposing the Enum for selecting the loss functions
    py::enum_<Loss>(m, "Loss")
        .value("MSE", Loss::MSE)
        .value("RMSE", Loss::RMSE)
        .export_values();

    py::enum_<nn::loss::LossFns>(m, "LossFns")
        .value("MSE", nn::loss::LossFns::MSE)
        .value("RMSE", nn::loss::LossFns::RMSE)
        .export_values();

    // define a CUDA submodule
    py::module_ cuda_functions = m.def_submodule("cuda", "Submodule for CUDA functions");
    // CUDA exposed attributes
    cuda_functions.def("cuda_prop", &implementation::printAttributes, R"pbdoc(
        Prints the essential information about shared memory for defining the matrix kernels
        for you GPUs compute capability.
    )pbdoc");
    cuda_functions.def("mmul", &cuda_interface::mmul, R"pbdoc(
        Performs matrix multiplication of two Eigen RowMajor floating point matrices
        :param A: RowMatrixXf
        :param B: RowMatrixXf
        :param C: a matrix multiplication algorithm to perform see MMulAlg for further
                  details
        The function also creates a special interface type called Fmatrix which is essentially
        all the requirements necessary to map a block of heap memory to an Eigen RowMatrixXf
    )pbdoc");
    py::enum_<matrix_kernels::MMulAlg>(cuda_functions, "MMulAlg")
        .value("SIMPLE", matrix_kernels::MMulAlg::SIMPLE)
        .value("SIMPLE2D", matrix_kernels::MMulAlg::SIMPLE2D)
        .value("TILED1D", matrix_kernels::MMulAlg::TILED1D)
        .value("TILED2D", matrix_kernels::MMulAlg::TILED2D);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}