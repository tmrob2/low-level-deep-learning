#pragma once

#include <iostream>
#include <cublas_v2.h>

namespace vector_kernels 
{
int vecAdd(float* A, float* B, float* C, int n);
} // vector kernels

namespace matrix_kernels 
{
enum MatrixLayout {
    ROWMAJOR,
    COLMAJOR
};

enum MMulAlg {
    SIMPLE,
    SIMPLE2D,
    TILED1D,
    TILED2D
};

/// @brief A class for converting between Eigen RowMajor matrices and device representations of 
/// those matrices
class FMatrix {
public: 
    FMatrix(float* data_, int rows_, int cols_): 
        data(data_), rows(rows_), cols(cols_), 
        size(rows_ * cols_ * sizeof(float)), dData(NULL),
        handleCreated(false) {}
    int toDevice();
    int copyToHost();
    int getRows() { return rows; }
    int getCols() { return cols; }
    float* getData() { return data; }
    float* getDeviceData() {return dData; }
    float* getOnes() { return ones; } // returns the device vector of ones
    void setData(float* data_) { data = data_; }
    size_t getSize() { return size; }
    float* data;
private:
    bool handleCreated;
    int rows;
    int cols;
    size_t size;
    float* dData;
    float* ones; // A vector of ones - this is a helper vector to perform columnwise summation
};
}

namespace implementation {

int printAttributes();
} // implementation