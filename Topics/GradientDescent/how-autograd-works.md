Neural networks use the gradient descent algorithm to update the weights and biases. Backpropagation is the fast algorithm to calculate the gradients. Backpropagation is based on the chain rule.

In Pytorch (also other deep learning tooks like tensflow, mxnet), it alread implements the auto gradient calculation engine, which could automatically track and calculate the gradient value and update the parameters (i.e. weights and biases).

# How does the autograd works internal?
We could use the mathmatically approach to deriviate the gradient descent in formula approach and know how it works in theory. When implementing in the code level, how could it be able to automatically do the grad computation for arbitrary architectures?

This learning note will take the pytorch as example to understand how it works internally even different framework might be slightly different.

## Data Container: Tensor
Tensor in Pytorch is similar as the ndarray array in numpy, which could represent any dimensions of data.

In the data structure, it has several important properteis:
- `data`: the variable holds the values
- `grad`: records the gradient value
- `grad_fn`: indicates how to calculate the gradient (deriviation)
- `is_leaf`: indicates whether the current node is a leaf node
- `requires_grad`: whether it needs to track the history gradients

## Gradient calcuation kernel: Function 
The `Function` is the kernel to calculate the gradient. Basically, in the neural network, the matrix compulations uses different ops to construct the model. Those ops is already built-in pytorch and could meet almost all the needs of the modeling.

`Function` is the base class that define the interface about how to calcualte the gradient via the api `backward`. And in pytorch, it already defines all the [backward functions](https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/symbolic_script.cpp) of the ops, like add, mul, div, clamp etc.

So, for each op, it will have a corresponding backward method. That is the core magic of how the autograd calculation implemented.

## Computation Graph
> [TODO]

## Resource
- [PyTorch Autograd: Understanding the heart of PyTorch’s magic](https://towardsdatascience.com/pytorch-autograd-understanding-the-heart-of-pytorchs-magic-2686cd94ec95)
- [PyTorch Autograd Explained - In-depth Tutorial](https://www.youtube.com/watch?v=MswxJw-8PvE)
- [Getting Started with PyTorch Part 1: Understanding how Automatic Differentiation works](https://towardsdatascience.com/getting-started-with-pytorch-part-1-understanding-how-automatic-differentiation-works-5008282073ec)
- [Automatic differentiation in PyTorch](https://openreview.net/pdf?id=BJJsrmfCZ)