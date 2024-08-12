cmake_minimum_required(VERSION 3.22)
project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)
#project(example)

find_package( PkgConfig )
pkg_check_modules( EIGEN3 REQUIRED eigen3 )
include_directories( ${EIGEN3_INCLUDE_DIRS} include)

find_package(OpenMP REQUIRED)

add_subdirectory(pybind11)
#pybind11_add_module(example example.cpp)


# Find the module development requirements (requires FindPython from 3.17 or
# scikit-build-core's built-in backport)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
#find_package(pybind11 CONFIG REQUIRED)

# Add a library using FindPython's tooling (pybind11 also provides a helper like
# this)
python_add_library(_core MODULE src/interface.cpp
                                src/activation.cpp
                                src/comp_graph.cpp 
                                src/matrix_functions.cpp 
                                src/linear_regression.cpp
                                src/loss.cpp WITH_SOABI)
                                
target_link_libraries(_core PRIVATE pybind11::headers OpenMP::OpenMP_CXX)

# This is passing in the version as a define just as an example
target_compile_definitions(_core PRIVATE VERSION_INFO=${PROJECT_VERSION})

# The install directory is the output (wheel) directory
install(TARGETS _core DESTINATION cppapi)