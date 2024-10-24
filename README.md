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
    Operation() {}
    std::shared_ptr<RowMatrixXf> _forward(std::shared_ptr<RowMatrixXf> input);
    std::shared_ptr<RowMatrixXf> _backward(std::shared_ptr<RowMatrixXf> output_grad);
protected:
    virtual void _output() = 0; // this is something that will be implemented when we have a ParamOperation
    virtual void _inputGrad(std::shared_ptr<RowMatrixXf> outputGrad) = 0;
    std::shared_ptr<RowMatrixXf> input_;
    std::shared_ptr<RowMatrixXf> output_;  
    std::shared_ptr<RowMatrixXf> input_grad_;
};
```

On the other hand, the `ParamOperation` class inherits from the `Operation` class implementing a few extra members and methods. The `ParamOperation` class instantiates some parameter matrix on its construction and also specifically implements `backward` (overriding the `Operation::backward`) method.
```c++
class ParamOperation: public Operation {
/*
The ParamOperation extends on the Operation class but accepts a paarameter in its
constructor.
*/
public:
    ParamOperation(std::shared_ptr<RowMatrixXf> param): Operation(), param_(param), param_grad_(nullptr) {}
    std::shared_ptr<RowMatrixXf> _backward(std::shared_ptr<RowMatrixXf> outputGrad);
    friend class Layer;
protected:
    virtual void paramGrad(std::shared_ptr<RowMatrixXf> outputGrad) = 0;
    std::shared_ptr<RowMatrixXf> param_; // Param is the forward prediction of the layer
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
    Layer(int neurons): neurons_(neurons) {};
    void _forward(std::shared_ptr<RowMatrixXf> input);
    void _backward(std::shared_ptr<RowMatrixXf> output_grad);
protected:
    virtual void setupLayer(std::shared_ptr<RowMatrixXf> input) = 0; 
    void _paramGrads();
    void _params(); 
    int neurons_;
    int param_operations = 0;
    std::vector<std::shared_ptr<RowMatrixXf>> params_;
    std::vector<std::shared_ptr<RowMatrixXf>> param_grads_;
    std::vector<std::shared_ptr<Operation>> operations_;
    std::vector<std::shared_ptr<Operation>> reversed_operations_;
    // cached operation variables
    // This gets a little tricky but I think the input will be shared with the neural network
    // class itself when we finally get to programming this
    // The nerual network will interface with the FFI and then everything will be shared from this
    std::shared_ptr<RowMatrixXf> input_; 
    std::shared_ptr<RowMatrixXf> output_;
    std::shared_ptr<RowMatrixXf> input_grad_;
};
```

The `Layer` class can then be specialised to create specific neural network layers. For example, the fully connected or dense layer can be implemented in the following way:
```c++
class Dense: public Layer {
public:
    Dense(int neurons, std::shared_ptr<Operation> activation): Layer(neurons) {}
protected:
    void setupLayer(std::shared_ptr<RowMatrixXf> input) override;
};
```

While this implementation style assists with easily creating new layer constructs, such as CNN or LSTM layers, it does abstract away from the underlying operations and is therefore a trade-off for development speed versus readability.

The loss abstract class for neural networs is implemented as below:
```c++
class Loss {
public:
    Loss() {}
    float forward(std::shared_ptr<RowMatrixXf> prediction, std::shared_ptr<RowMatrixXf> target);
    RowMatrixXf backward();
protected:
    virtual float _output() = 0;
    virtual RowMatrixXf _inputGrad() = 0;
    std::shared_ptr<RowMatrixXf> prediction_;
    std::shared_ptr<RowMatrixXf> target_;
    RowMatrixXf input_grad_;
};
```

## The Neural Network Class

The neural network class contains the following:
1. A list of ```Layer``` classes as an attribute. Layers need to be predefined such as ```Dense``` and have ```_forward``` and ```_backward``` methods.
2. Each layer has a list of ```Operation``` classes which it performs and this is done using the ```Layer::setupLayer``` method. 
3. The ```Operation``` classes included in the ```Layer``` have ```_forward``` and ```_backward``` methods as well.
4. In each operation the shape of the ```output_grad``` which is a ```std::shared_ptr<RowMatrixXf>``` input into ```Operation::_backward``` must be the same matrix shape as the ```Layer::output_``` attribute. The same is true for the shapes of the ```input_grad``` passed backward in the ```Layer::_backward``` method and the ```Layer::input_``` attribute.
5. Some operations have neural network parameters (stored in the ```Layer::params_``` atrtribute). These operations inherit from the ```ParamOperation``` class. The same constraint on the input and output shapes apply to layers in their forward and backward methods. 
6. A Neural network will also have a ```Loss```. This class will take output of the last operation from the neural network and the target. We have to check that their shapes are the same and calculate both a loss value (scalar) and the ```loss_grad``` that will be fed into the output layer when starting backpropagation.  

Before discussing how the Neural network works, a note on shared pointers between Pybind11 and c++.
To work with abstract classes, i.e. those classes which implement ```virtual``` methods we need a 
way of describing them to Python. A definition of the interface for the ```Operation``` class is now given. As can be observed above - the ```Operation``` class contains virtual methods which are instantiated for specific operations. To describe this behaviour we first require a ```PyOperation``` class which describes every virtual method in the class. 
```c++
class PyOperation: public nn::Operation {
public:
    using nn::Operation::Operation;
    void _output() override {
        PYBIND11_OVERRIDE_PURE(
            void,
            Operation,
            _output,
        );
    }

    void _inputGrad(std::shared_ptr<RowMatrixXf> outputGrad) override {
        PYBIND11_OVERRIDE_PURE(
            void,
            Operation,
            _inputGrad,
        );
    }
};
```

The interface can then contain a reference to the virtual ```Operation``` but this should never be called as it's behaviour is not completely defined. Here in some trickery the py class ```Operation``` inherits from ```PyOperation``` which does have all of its methods defined - albeit with placeholders. 
```c++
py::class_<nn::Operation, PyOperation, std::shared_ptr<nn::Operation>>(m, "Operation")
        .def(py::init<>());
```

Once we have defined the intefaces python operation class we can define specific ```Operation```s. The Python interface ```Sigmoid``` class is described to inherit from the ```Operation``` abstract class which is fully defined because of ```PyOperation```. We also say that ```Sigmoid``` requires description for shared pointer behaviour between Python and C++. A note of caution of ownership here. In the instantiation of this class, both C++ and Python will have ownership of the shared pointer but the ```Sigmoid``` class will be owned by Python. This means that we will be potentially referencing back to the Python memory addresses which could result in undesired behaviour. Therefore, for the ```NeuralNetwork``` class it is best to take ownership of any attributes in C++. This can be done with ```std::move```. An example of the specific implementation of an ```Operation``` with the ```Sigmoid``` class: 
```c++
py::class_<nn::activation::Sigmoid, nn::Operation, std::shared_ptr<nn::activation::Sigmoid>>(m, "Sigmoid")
    .def(py::init<>());
```
