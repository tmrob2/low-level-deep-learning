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

# development code

This is research code.

The aim is not replace PyTorch or TensorFlow but to understand the operations and scalability from scratch. Errors are not handled nicely :(.
