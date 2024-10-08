The purpose of this code is to study facets of deep learning. In particular:
1. Scalability of operations from CPU, multiprocessing CPU, naive GPU, high throughput GPU:
    - Where are the boundaries? When should certain operations be done on the CPU vs GPU and are there times when CPU is better for deep learning than GPU. Does it all just depend on the matrix operations?

# low-level-deep-learning
deep learning...the hard way with cpp

Ideas of things of interest - parallel processing of linear regression and neural networks
from the ground up. That is:
1. sequential compute
2. parallel cpu compute with varying number of threads - understand the runtime as a function of threads for different problem sizes.
3. gpu compute

## Documentation

Make some documentation and put a link to it here. 

# Development

This is research code.

The aim is not replace PyTorch or TensorFlow but to understand the operations and scalability from scratch. Errors are not handled nicely :(.

The code base is divided into basics, located in `src/tutorial`, and the general neural network library, located in `src/nn`. Basics are useful to understand the premise of this library including: TODO
- basics lib point 1
- etc
- Demonstrate that the library is built on implementation of Eigen::Matrix<...> row-major floating point matrices and Eigen operations
- How does the cuda implementation work with the Python interface and communication with RowMajorXf matrix -> pointers

On the other hand, things really start to get interesting in the general neural network implemnentation. Here, the focus is all about neural network operations, their implementation device strategy, and the corresponding necessary memory operations for computational efficiency. 

For example, the `Operation` abstract class which has methods such as `forward` and `backward`, which implement the forward pass and backward pass of the neural network respectively. The foreign interface is implemented in python, therefore a `numpy` floating-point (f32) array object is instantiated and moved (ownership) to the C++ API. To avoid copies, the `Operation` class uses smart pointers and therefore nerual network operations are done on pointers to Eigen row-major f32 matrices seen in the snippet below:
```c++
// Operation class
// --------------------
// Stores the input and calls output
std::shared_ptr<RowMatrixXf> Operation::forward_(std::shared_ptr<RowMatrixXf> input) {
    input_ = input; // stores the shared pointer in the Operation class
    // Calls the abstract output_fn() and must be implemented for
    // specific implementations fo Operation
    output_ = std::make_shared<RowMatrixXf>(RowMatrixXf(output_fn())); 
    return output_;
}
```

The fundamentals of the neural network are the `Operation` and `ParamOperation` classes.
The purpose of the `Operation` class is to handle in-memory matrix operations in an efficient manner.
In particular, this means that we don't want to be copying matrices all over the place.
Because deep learning is essentially matrix operations - the more that we copy matrices
the worse the peformance will be. 
Neural network layers are a series of operations followed by a non-linear operation.
- For example, it might be the weight matrix multiplication followed by a bias addition.
- This could then be followed by an activation function (non-linear operation) such as sigmoid.

The `Operation` class has the following structure: 
```c++
class Operation {
public:
    Operation(): input_(nullptr), output_(nullptr), input_grad_(nullptr) {}
    std::shared_ptr<RowMatrixXf> forward_(std::shared_ptr<RowMatrixXf> input);
    Eigen::Ref<RowMatrixXf> backward(Eigen::Ref<RowMatrixXf> output_grad);
protected:
    virtual RowMatrixXf output_fn() = 0;
    //virtual RowMatrixXf input_fn() = 0;
    virtual RowMatrixXf inputGrad_(Eigen::Ref<RowMatrixXf> outputGrad) = 0;
    std::shared_ptr<RowMatrixXf> input_;
    std::shared_ptr<RowMatrixXf> output_; 
    std::shared_ptr<RowMatrixXf> input_grad_;
};
```

On the other hand, the `ParamOperation` class inherits from the `Operation` class implementing a few extra members and methods. The `ParamOperation` class instantiates some parameter matrix on its construction and also specifically implements `backward` (overriding the `Operation::backward`) method.
```c++
class ParamOperation: public Operation {
public:
    ParamOperation(Eigen::Ref<RowMatrixXf> param_): Operation(), param_(param_), param_grad_(nullptr) {}
    Eigen::Ref<RowMatrixXf> backward(Eigen::Ref<RowMatrixXf> outputGrad);
    friend class Layer;
protected:
    virtual RowMatrixXf paramGrad(Eigen::Ref<RowMatrixXf> outputGrad) = 0;
    Eigen::Ref<RowMatrixXf> param_; // Param is the forward prediction of the layer
    std::shared_ptr<RowMatrixXf> param_grad_; // param grad is the partial derivative with repsect to the parameters of the layer
};
```

Finally, the `Layer` class is an extension of (parameter) operations and represents a layer of neurons in a neural network. `Layer` implements forward and backward methods consist of sending the input successively forward through a series of operations. We also implement the following bookkeeping operations:
1. Defining the correct series of operations in the `_setup_layer` function and initialising
   and storing all of the parameters in these `Operation`s.
2. Storing the correct vales in the `input_` and `output` forward method.
3. Performing the correct assertion checking in the backward method.
4. `params` and `params_grads` functions simply extract the parameters aand their gradients with
   respect to loss from the ParamOperations within the layer.

There is some `C++` trickery in the `Layer` implementation because `operations` could be either `ParamOperation`s or `Operation`s. We do not know which the layer will consist of at compile time so we need to be able to deal with both. Here we make use of a `vector` of `shared_ptr`s to `Operation`s and then work out which is which at runtime. A neat trick is also implemented which instead of versing a vector of smart pointers to `Operation`s we also instantiate the reversed smart pointers at the creation of the `operations` vector which is implemented in `O(1)`. 
There are some further implementation details to work out here: 
- How does the `vector` of `shared_ptr`s implementation work with the GPU?
```c++
class Layer {
public:
    Layer(int neurons_): neurons(neurons_){}
    RowMatrixXf forward(Eigen::Ref<RowMatrixXf> input);
    RowMatrixXf backward(Eigen::Ref<RowMatrixXf> output_grad);
protected:
    virtual void setupLayer(std::shared_ptr<RowMatrixXf> input) = 0;
    void paramGrads();
    void params();
    int neurons;
    // Caches
    std::shared_ptr<RowMatrixXf> input;
    RowMatrixXf output;
    bool first;
    std::vector<std::shared_ptr<RowMatrixXf>> params_;
    std::vector<std::shared_ptr<RowMatrixXf>> param_grads_;
    std::vector<std::shared_ptr<Operation>> operations;
    std::vector<std::shared_ptr<Operation>> reversed_operatations; 
};
```

The `Layer` class can then be specialised to create specific neural network layers. For example, the fully connected or dense layer can be implemented in the following way:
```c++
class Dense: public Layer {
public:
    Dense(int neurons, std::unique_ptr<Operation> activation_)
    : Layer(neurons), activation(std::move(activation_)) {}
protected:
    void setupLayer(std::shared_ptr<RowMatrixXf> input_) override;
    std::shared_ptr<Operation> activation;
};
```

While this implementation style assists with easily creating new layer constructs, such as CNN or LSTM layers, it does abstract away from the underlying operations and is therefore a trade-off for development speed versus readability.  