#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <Eigen/LU>

#include "activation.hpp"
#include "common_types.hpp"
#include "comp_graph.hpp"
#include "matrix_functions.hpp"
#include "linear_regression.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

// N.B. this would equally work with Eigen-types that are not predefined. For example replacing
// all occurrences of "Eigen::MatrixXd" with "MatD", with the following definition:
//
//  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatD;

// ----------------
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
        .def_readonly("N", &LinearRegression::N)
        .def_readonly("P", &LinearRegression::P)
        .def_readonly("W", &LinearRegression::W)
        .def_readonly("B0", &LinearRegression::B0)
        .def_readonly("dLdB", &LinearRegression::dLdB)
        .def_readonly("dLdW", &LinearRegression::dLdW);

    py::enum_<Activation>(m, "Activation")
        .value("SIGMOID", Activation::SIGMOID)
        .value("SQUARE", Activation::SQUARE)
        .value("LEAKY_RELU", Activation::LEAKY_RELU)
        .export_values();

    py::enum_<Loss>(m, "Loss")
        .value("MSE", Loss::MSE)
        .value("RMSE", Loss::RMSE)
        .export_values();

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}