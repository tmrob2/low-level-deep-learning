#pragma once

#include <iostream>

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

class FMatrix {
public: 
    FMatrix(float* data_, int rows_, int cols_): 
        data(data_), rows(rows_), cols(cols_), size(rows_ * cols_ * sizeof(float)) {}
    int mmul(FMatrix& B, FMatrix& C, MMulAlg alg);
    int getRows() { return rows; }
    int getCols() { return cols; }
    float* getData() { return data; }
    void setData(float* data_) { data = data_; }
    size_t getSize() { return size; }
    float* data;
private:
    int rows;
    int cols;
    size_t size;
};
} // matrix_kernels


namespace implementation {

int printAttributes();
} // implementation