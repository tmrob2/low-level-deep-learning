cmake_minimum_required(VERSION 3.22)
project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX CUDA)
#project(example)

find_package( PkgConfig )
pkg_check_modules( EIGEN3 REQUIRED eigen3 )
include_directories( ${EIGEN3_INCLUDE_DIRS} include)

find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)
add_subdirectory(pybind11)
#pybind11_add_module(example example.cpp)


# Find the module development requirements (requires FindPython from 3.17 or
# scikit-build-core's built-in backport)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
#find_package(pybind11 CONFIG REQUIRED)

# create a GPU library and then link this library to the python executable
set(LibHeaders
  include/cuda/cu_matrix_functions.h
)

set(LibSources
  src/cuda/cu_matrix_functions.cu
)

add_library(libcppapi-gpu STATIC ${LibSources} ${LibHeaders})
target_link_libraries(libcppapi-gpu PRIVATE CUDA::cudart CUDA::cublas)
set_target_properties(libcppapi-gpu PROPERTIES LINKER_LANGUAGE CUDA CUDA_ARCHITECTURES OFF
                      POSITION_INDEPENDENT_CODE ON)


# Add a library using FindPython's tooling (pybind11 also provides a helper like
# this)
python_add_library(_core MODULE src/interface.cpp
                   src/activation.cpp
                   src/comp_graph.cpp 
                   src/matrix_functions.cpp 
                   src/linear_regression.cpp
                   src/loss.cpp 
                   src/nn.cpp 
                   WITH_SOABI)
                                
target_link_libraries(_core PRIVATE pybind11::headers OpenMP::OpenMP_CXX libcppapi-gpu)

# This is passing in the version as a define just as an example
target_compile_definitions(_core PRIVATE VERSION_INFO=${PROJECT_VERSION})

# The install directory is the output (wheel) directory
install(TARGETS _core DESTINATION cppapi)