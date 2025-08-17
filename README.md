# MicroGrad

> This is a copy of the [Micrograd by Andrej Karpathy](https://github.com/karpathy/micrograd/). It was made with the intention of learning the inner workings of the original.

A tiny automatic differentiation engine and a small neural network library built in Python. It shows how backpropagation really works under the hood without relying on big libraries like PyTorch or TensorFlow. However, the structure of the code is in line with the PyTorch codebase.

### Functionalities
 - Implements a `Value` class that tracks data, gradients, and builds a
   computation graph automatically.
 - Supports reverse-mode automatic differentiation (`.backward()` does backprop).
 - Includes a small neural network API (`Neuron`, `Layer`, `MLP`) built on top of the autograd engine.
 - Can train a simple multi-layer perceptron (MLP) on toy datasets using gradient descent.

### Goal of the Project
This is meant as an **educational project**. The entire engine is only ~100 lines of code, so it’s easy to read and understand how autodiff and neural nets work at the lowest level.

### Example
```python
from micrograd.engine import Value
from micrograd.nn import MLP
# Build a tiny neural net 2 inputs, two hidden layers [16, 16], and 1 output
model = MLP(2, [16, 16, 1])
# Example forward pass
x = [Value(1.0), Value(-2.0)]
y_pred = model(x)
print(y_pred)
```

### Testing
This project includes a small test suite to make sure the autograd engine works correctly. The tests check that gradients from micrograd match those from PyTorch on the same operations.
> [!important]
> Install PyTorch using relevant installation from https://pytorch.org/get-started/locally/

To run the tests:
```bash
python -m pytest
```
If everything is correct, you should see all tests pass.

### Inspiration
Based on [Andrej Karpathy’s original micrograd](https://github.com/karpathy/micrograd).
