## Deep Learning

***1. What is Deep Learning?***
> The deep in deep learning isn’t a reference to any kind of deeper understanding achieved by the approach; rather, it stands for this idea of successive layers of representations. 

*Neural Networks* is the models most used in layered representation, which are structured in literal layers stacked on top of each other (layer by layer connecting the neural networks). 

*Weights* of a layer specifies what the layer do to the input and the transformation of a layer is implemented by the parameters with the weights. So, in deep learning, the *learning* is finding a set of values for weights for all layers to get the predictions close to the expected output. 

As mentioned, machine learing needs have a way to measure how good the training is. Same in deep learing, there is *loss function (or objective function)* measures the distance between the prediction and the expected target (how good/bad it is).

Deep learning leverage the signal from loss function and continuously adjust the weights to the direction of reducing the loss. The adjustment is call *optimizer* which supervises how to perform the learning process. The basic and popular algorithm is *Backpropagation*.

*Training loop* repeats a sufficient number of times (epochs) go over the same process against the trainig data set. 

***2. Why Deep Learing?***
1. **Feature automatically learning**: Usually deep learning has better performance, and the most important thing is the it automates the most crucial and difficult step in ML: feature engineering.
2. **Learn all layers of representation jointly**, rather than in succession (greedily).
3. Two other characteristics how learn from data:
    - Incremental, layer-by-layer way in which increasingly complex representations are developed
    - Intermediate incremental representations are learned jointly

***3. Why Deep Learning Works Right Now?***
- **Hardware**
- **Dataset and Benchmarks**
- **Algorithm advances**
    - Better *activation function* for neural layers
    - Better weight-initializations schemes
    - Better optimization schemes, such as RMSProp and Adam

## Mathmatical basic for neural networks

### 1. Three basic things for neural network training
- **A loss function** - how network measure its performance on training data, which used to guide the right direction
- **An optimizer** - how the network adjust/update itself based on the data and the loss function
- **Metries to monitor during training and test** - the quality of the training result

### 2. Data representation neural networks

#### Tensors

**Definition of tensors**
*Tensors* - multidimensional numpy arrays (almost always numerical data). Tensors are a generalization of matrices to an arbitrary number of dimensions. Dimension is often called an axis in the context of tensors. Dimensionality can denote either the number of entries along a specific axis or the number of axes in a tensor. The rank of a tensor is the number of axes, like a *tensor of rank n*.
    - Scalars (0D tensors), only one number
    - Vectors (1D tensors), an array of numbers
    - Matrices (2D tensors), an array of vectors
    - 3D tensors or higher-dimentional tensors

**Key attributes of tensors**
- Number of axes (rank)
- Shape - the tuple of integers that describe how many dimensions the tensor has along each axis, like a matrix has shape (3,5)
- Data type (usually called dtype in python library)

### 3. Tensors operations - The gears of neural network

All transformations learned by deep learning NN can be reduced to tensor operations applied to tensors of numeric data.

1. Element-wise operation

Element-wise operations are applied independently to each entry in the tensors. Activation function like relu and addition are element-wise. That means it could be parallel implemented.

2. Broadcasting

Smaller tensor will be broadcasted to matched the shape of the larger tensor, when different shapes of two tensors be added. Broadcasting consists of two steps:
- Axes (called broadcast axes), are added to smaller tensor to match the ndim of larger tensor
- Smaller tensor is repeated alongside these new axes to match the full shape of the larger tensor

3. Tensor dot

Tensor dot operation is also called tensor product. Contray to element-wise operations, it combines entries in the input tensors.

4. Tensor reshaping

Reshaping a tensor means rearranging its rows and columns to match a target shape. The reshaped tensor has the same total number of coefficients as the initial tensor.

**Data batches**
In general, the first asix in all data tensors in deep learning will be the *sample axis (i.e. sample dimension)*. In addition, DL model don't process an entire dataset, but break it into small batches. When considering a batch tensor, the first axis is called *batch axis or batch dimension*。

**Example**
- Vector data - 2D tensors of shape (samples, features)
- Timeseries data or sequence data - 3D tensors of shape (sample, timesteps, features)
- Images - 4D tensors of shape (samples, height, width, channels)
- Video - 5D tensors of shape (sample, frames, height, width, channels)


### 4. Gradient-based optimization - the engine of neural networks

*Weights (trainable parameters)* contains the infor learned by network from the training data. Initially, weight matrices are filled with small random value (the step called random initialization).

*Training* the gradual adjustment of weights, it is basically machine learing all about. It happens within a training loop. Each iteration over the training data is called an *epoch*.

*How to increase/decrease the coefficient for learning? And by how much?*

- Naive solution is freeze all weights except one scalar coefficient and try different values for it.
- Since the operation is differentiable, compute the gradient of the loss with regard to the network's coefficients.

A gradient is the derivative of a tensor operation. Algorithms like Stochastic Gradient Descent, mini-batch SGD, SGD with momentum, Adagrad, RMSProp etc, they are known as *optimization methods* or *optimizers*. Why momentum with SGD? It resolves the convergence speed and local minima problems.

*Backpropagation algorithm (i.e. reverse-mode differentiation)*: applying the chain rule ( *f(g(x)))' =f'(g(x))*g'(x)* ) to compute the gradient values of a neural network. Modern frameworks have the capability of symbolic differentiation, which given a chain of operations with a known derivative they can compute a gradient function for the chain (by applying the chain rule).

### 5. Core components in neural network

The objects in nerual network:
- *Layers*, which are combined into a network (or model)
- The *input data* and corresponding *targets*
- The *loss function*, which defines the feedback signal used for learning. It is the quantity that will be minimized during training.
- The *optimizer*, which determines how learning proceeds (i.e. how the network be updated based on the loss function)

A deep-learning model is a directed, acyclic graph of layers. The topology of a network defines a hypothesis space. By choosing a network topology, it constrains the space of possibilities to a specific series of tensor operations, mapping input data to output data.

The neural network has multiple output might have multiple loss functions (one per output), but the gradient-descent process must be based on a single scalar loss value.