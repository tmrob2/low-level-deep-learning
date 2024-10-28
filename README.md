# Low Level Deep Learning
Deep learning from scratch...the hard way with C++.

Project aims: 
- Starting with hard-coded dense neural network compute all of the operations of a two layer network.
- Construct a generalised neural network framework that is able to perform any layer operations.
- Work our way up from dense, CNN, LSTM, transformer to constucting a LLM from scratch in with both CPU and GPU computing capabilities.  

## Research Questions
The purpose of this code is to study facets of deep learning. In particular:
1. Scalability of operations from CPU, multiprocessing CPU, naive GPU, high throughput GPU:
    - Where are the boundaries? When should certain operations be done on the CPU vs GPU and are there times when CPU is better for deep learning than GPU. Does it all just depend on the matrix operations?


## Research Objectives
Ideas of things of interest:
1. sequential compute - data structures and memory analysis
2. parallel cpu compute with varying number of threads:
    1. Are different data structures required? Hypothesis is not because NN are implicitly sequential computation data structures and the best we can aim for is faster matrix multiplication.
    2. Examing the runtime as a function of threads for different problem sizes - what does the runtime optimisation curve look like
    3. Observe compute phenomena to examine bottlenecks in this process.
3. GPU compute

## Development
This is research code.

The aim is not replace PyTorch or TensorFlow but to understand the operations and scalability of all operations of the neural network from scratch. Errors are not handled nicely :(.

The code base is divided into basics, located in `src/tutorial`, and the general neural network library, located in `src/nn`. Basics are useful to understand the premise of this library including: TODO
- basics lib point 1
- etc
- Demonstrate that the library is built on implementation of Eigen::Matrix<...> row-major floating point matrices and Eigen operations
- How does the cuda implementation work with the Python interface and communication with RowMajorXf matrix -> pointers

On the other hand, things really start to get interesting in the general neural network implemnentation. Here, the focus is all about neural network operations, their implementation device strategy, and the corresponding necessary memory operations for computational efficiency. 

### Neural Network Operations


![Neural Network Operation Schematic](./images/deep-learning-operation.drawio.pdf)



