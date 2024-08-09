#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <Eigen/LU>

#include "activation.hpp"
#include "common_types.hpp"
#include "comp_graph.hpp"

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

    py::enum_<Activation>(m, "Activation")
        .value("SIGMOID", Activation::SIGMOID)
        .value("SQUARE", Activation::SQUARE)
        .value("LEAKY_RELU", Activation::LEAKY_RELU)
        .export_values();

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}