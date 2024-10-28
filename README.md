# Low Level Deep Learning
Deep learning from scratch...the hard way with C++.

Project aims: 
- Starting with hard-coded dense neural network compute all of the operations of a two layer network.
- Construct a generalised neural network framework that is able to perform any layer operations.
- Work our way up from dense, CNN, LSTM, transformer to constucting a LLM from scratch in with both CPU and GPU computing capabilities.  

## Simple Example Usage

The framework is best accessed directly from Python using Pybind11, however, can be used directly as a C++ API. 
```Python
import cppapi
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

diabetes = load_diabetes()
targets = diabetes.target.astype(np.float32)
data = diabetes.data.astype(np.float32)

# Normalise the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
targets_ = targets.reshape(-1, 1)

# make a train test split
X_train, X_test, Y_train, Y_test = \
    train_test_split(data_scaled, targets_, test_size=0.2, random_state=42)

neurons = 1
lr = 0.01
epochs = 10
eval_every = 10
batch_size = X_train.shape[0]
restart = True
verbose = 2

nn = cppapi.NeuralNetwork2([
        cppapi.Layer2(cppapi.LayerType.Dense, 16, cppapi.ActivationType.Sigmoid),
        cppapi.Layer2(cppapi.LayerType.Dense, 16, cppapi.ActivationType.Sigmoid),
        cppapi.Layer2(cppapi.LayerType.Dense, 1, cppapi.ActivationType.Linear)
    ], 
    cppapi.LossType.MSE)

trainer = cppapi.Trainer(nn, cppapi.OptimiserType.SGD, lr)

trainer.fit(X_train, Y_train, X_test, Y_test, epochs, eval_every, batch_size, restart, verbose)
```

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
This is a research framework.

The aim is not replace PyTorch or TensorFlow but to understand the operations and scalability of all operations of the neural network from scratch. Errors are not handled nicely :(.

The code base is divided into basics, located in `src/tutorial`, and the general neural network library, located in `src/nn2`. Basics are useful to understand the premise of this library including: 
- Demonstrate that the library is built on implementation of Eigen::Matrix<...> row-major floating point matrices and Eigen operations
- How does the cuda implementation work with the Python interface and communication with RowMajorXf matrix -> pointers
- Demonstrating the mechanics of forward/backward propagation through the 

On the other hand, things really start to get interesting in the general neural network implemnentation. Here, the focus is all about neural network operations, their implementation device strategy, and the corresponding necessary memory operations for computational efficiency. 

### Neural Network Operations

The layers of neural networks can be considered as a series of operations which is just some differentiable function with an input and and output. The general idea of the operation can be summarised in the figure below. We have some input data into the operation and then compute ```output = f(x)``` where ```x``` is some data input. The parameter is how we control the optimisation of the neural network to approximate some function. For example, linear regression would just be a parameter ```W``` and ```Op``` would be ```mmul(X, W)```. This would then be followed by a bias operation which would be ```mmul(X, W) + beta```. Finally, we compute loss between forward pass and the supervised target using some loss function. In the backpropagation, we first differentiate each of the nueral network operations and then work backwards through each of them until we compute the entire chain of functions a.k.a. the chain rule.  

<div align="center">
 <img src="./images/deep-learning-operation-2.jpg", alt="Alt text">
</div>

So the question is how to represent this in a memory efficient manner. As a bit of context, I had originally implemented the neural network framework using class inheritence but this become unmanagement in terms of runtime and also effectively understanding how to debug.


