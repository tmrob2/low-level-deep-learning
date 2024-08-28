#include "cuda/cu_matrix_functions.h"
#include <cuda.h>

#define CHECK_CUDA(func)                                                              \
{                                                                                     \
    cudaError_t status = (func);                                                      \
    if (status != cudaSuccess) {                                                      \
        printf("CUDA API failed at line %d with error %s (%d)\n",                     \
                __LINE__, cudaGetErrorString(status), status);                        \
        return EXIT_FAILURE;                                                          \
    }                                                                                 \
}                                                                                     \

#define TILE_WIDTH 16                                                                 \

__global__ void vectorAdd(const float*A, const float* B, float* C, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

__global__ void simpleMMul2D(const float** A, const float** B, float** C, int blockWidth) {
    // using 2d coordinates for the matrix
    // TODO how do we specify a 2D matrix
    // We assume that the block is N x N
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((row < blockWidth) && (col < blockWidth)) {
        float pValue = 0.f;
        for (int k = 0; k < blockWidth; ++k) {
            pValue += A[row][k] * B[k][col];
        }
        C[row][col] = pValue;
    }
}

__global__ void simpleMMul1D(const float* A, const float* B, float* C, int blockWidth) {
    // Because of the compute-to-global-memory ratio of this kernel the like occupancy 
    // will be around 2%
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row < blockWidth) && (col < blockWidth)) {
        float pValue = 0.f;
        for (int k = 0; k < blockWidth; ++k) {
            pValue += A[row * blockWidth + k] * B[k * blockWidth + col];
        }
        C[row * blockWidth + col] = pValue;
    }
}

__global__ void tiledSquareMMul(const float* A, const float* B, float* C, int blockWidth) {
    // All threads in a block can acess 
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    // need the x, y direction block location
    int bx = blockIdx.x; int by = blockIdx.y;
    // need the x, y direction thread locations in the block
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * blockWidth + ty;
    int col = bx * blockWidth + tx;

    // We first need to load the data into shared memory
    // count out the phases needed to compute all of the dot products
    float pValue = 0.f;
    for (int ph=0; ph < blockWidth / TILE_WIDTH; ++ph) {
        // load the data into shared memory
        // The portion of the matrix A needed will depend on the tileing phase
        // of the computation.
        Mds[ty][tx] = A[row * blockWidth + TILE_WIDTH * ph + tx];
        Nds[ty][tx] = B[(ph * TILE_WIDTH + ty) * blockWidth + col];
        // Doesn't affect the for loop within the kernel just the threads involved in 
        // the shared memory within the block.
        // We don't want the theads overwriting the content of shared memory
        // until we are finished using it.
        __syncthreads();
        // After the above syncthreads all of the data will be loaded into SM
        // Now compute the partial dot product of tx,ty
        for (int k = 0; k < TILE_WIDTH; ++k) {
            pValue += Mds[ty][k] * Nds[k][tx];
        }
        // make sure that all of the inner products have been computed before
        // starting to overwrite the data in shared memory
        __syncthreads();
    }
    C[row * blockWidth + col] = pValue;
}

__global__ void tiledSquareMMul2D(const float** A, const float** B, float** C, int blockWidth) {
    // setup a tile of shared memory
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    // get the x,y direction of the block
    int bx = blockIdx.x; int by = blockIdx.y;
    // get the x, y direction of the thread within the block
    int tx = threadIdx.x; int ty = threadIdx.y;

    // compute the row and the column of the C matrix to be computed
    int row = by * blockWidth + ty;
    int col = bx * blockWidth + tx;

    float pValue = 0.f;
    // count out the phases necessary to compute the complete inner product.
    for (int ph = 0; ph < blockWidth / TILE_WIDTH; ++ph) {
        // use the threads to load the data into shared memory
        Mds[ty][tx] = A[row][ph * TILE_WIDTH + tx];
        Nds[ty][tx] = B[ph * TILE_WIDTH + ty][col];
        // synchronise the threads so that all of the data has been loaded into 
        // shared memory before computing the partial
        __syncthreads();
        // loop over all of the elements with the tile
        for (int k = 0; k < TILE_WIDTH; ++k) {
            pValue += Mds[ty][k] * Nds[k][tx];
        }
        // make sure that all of the inner products have been computed
        // before we start overwriting the data in shared memory
        __syncthreads();
    }
    C[row][col] = pValue;
}

namespace vector_kernels {

int vecAdd(float* A, float* B, float* C, int n) {
    /*
    Assuming that the host input memory has already been alocated.
    A is an input vector
    B is an input vector
    C is an output vector that holds the memory allocated to the solution
    */
    size_t size = n * sizeof(float);

    float* dA = NULL;
    CHECK_CUDA(cudaMalloc((void**)&dA, n))
    
    float* dB = NULL;
    CHECK_CUDA(cudaMalloc((void**)&dB, n))

    float* dC = NULL;
    CHECK_CUDA(cudaMalloc((void**)&dC, n))

    // Copy the data from the host to the device vectors
    CHECK_CUDA(cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dB, B, size, cudaMemcpyHostToDevice))

    // perform the kernel operation

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(dA, dB, dC, n);

    // copy the device memory for dC over to C
    CHECK_CUDA(cudaMemcpy(C, dC, size, cudaMemcpyDeviceToHost))

    // shut everything down and clean up 
    CHECK_CUDA(cudaFree(dA))
    CHECK_CUDA(cudaFree(dB))
    CHECK_CUDA(cudaFree(dC))
    return 0;
}
}

namespace matrix_kernels 
{

int FMatrix::mmul(FMatrix& B, FMatrix& C, MMulAlg alg) {
    // computes the matrix product of self and B using the algorithm 
    // defaulted to SIMPLE and collects the results in C

    // Allocate the device memory for A, B, C
    float* dA = NULL;
    CHECK_CUDA(cudaMalloc((void**) dA, this->size))

    float* dB = NULL;
    CHECK_CUDA(cudaMalloc((void**) dB, B.getSize()))

    float* dC = NULL;
    CHECK_CUDA(cudaMalloc((void**) dC, C.getSize()))

    // copy the data from the host to the device
    CHECK_CUDA(cudaMemcpy(dA, this->getData(), this->size, cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dB, B.getData(), B.getSize(), cudaMemcpyHostToDevice))

    // perform the kernel operation
    int threadsPerBlock = 1024;


    switch (alg) {
        case MMulAlg::SIMPLE:
            break;
        case MMulAlg::SIMPLE2D:
            break;
        case MMulAlg::TILED1D:
            break;
        case MMulAlg::TILED2D:
            break;
        default:
            std::cout << "Matrix multiplication method not implemented\n";
            break;
    }

    // copy the device memory for dC over to C
    CHECK_CUDA(cudaMemcpy(C.data, dC, size, cudaMemcpyDeviceToHost))

    // shut everything down and clean up 
    CHECK_CUDA(cudaFree(dA))
    CHECK_CUDA(cudaFree(dB))
    CHECK_CUDA(cudaFree(dC))
    return 0;
}

}

namespace implementation {

int printAttributes() {
    int devCount;
    CHECK_CUDA(cudaGetDeviceCount(&devCount))
    for (int i=0; i < devCount; ++i) {
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties_v2(&prop, i));
        int max_threads = prop.maxThreadsPerBlock;
        int shared_mem = prop.sharedMemPerBlock;
        std::cout << "max threads per block: " << max_threads << std::endl;
        std::cout << "shared memory per block: " << shared_mem << " bytes" << std::endl;
    }
    return 0;
};
}