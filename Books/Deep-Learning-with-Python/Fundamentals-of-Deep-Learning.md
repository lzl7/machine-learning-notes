# Relationship between: Artificial Intelligence, Machine Learning, Deep Learning

## Basic Definitions
**Artificial Intelligence (AI)** refers to the effort to automate intellectual tasks normally performed by humans.

**Machine Learning**: is the algorithms to learn without explicitly programmed. 

**Deep  Learning**: is a special subfield of machine learning, it emphases on the successive layers of increasing learning.

Hence, AI includes the machine learning as well as deep learning, and also other approaches that even without any learning like handcraft programmed rules.

## Evolutions

### Artificial Intelligence Early Age
- AI born in 1950s
- *Symbol AI* (1950s-1980s, No Learning): handcraft rules
    - Expert Systems: 1980s

Symbol AI is suitable to solve well-defined logical problems. However, it barely helps on the complex fuzzy problems. So, the new approach - *Machine learning* comes out.

### Machine Learning (ML)

Machine Learning started flourish in 1990s, it tries to achieve:
- Beyond knowing perform what programmer order, but learn to perform specified tasks.
- Automatically learn the rules from the data instead of only receiving handcraft rules from programmer

***1. What is the difference between ML and classical programming?***

ML is *trained* rather than *explictly programmed*. The comparison between classic programming and ML:
- Classical programming:
    - Input: data and rules
    - Output: *answers*
- Machine learning:
    - Input: data and answers
    - Output: *rules*

***2. What does machine learning do?***

Machine learning is learning the representation of data. There are three key things for machine learning:
- **Input data points**: the input resource used for training 
- **Expected output examples**: the labelled data as the ground truth for training prediction 
- **Approach to measure the algorithm performance (how good)**: measure the difference between the prediction and the expected output, this measurement provides the feedback signal to algorithm to adjust the training. The adjustment step is called *learning*.

The central problem in machine learning is all about representation, i.e. *meaningfully transform data*. Representation is a diffferent way to present or encode the input data.

***Learning*** is an automatic search process for better representation in machine learning. Machine learning algorithms aren't usually creative finding the transformations, but merely search through the predefined set of operations (called *hyphothesis space*).

> **Machine learning is, technically: searching for useful representations of some input data, within a predefined space of possibilities, using guidance from a feedback signal.**

***3. History of Machine learning***

1. **Probabilitic Modeling**, apply statistic to data analysis
    - Naive Bayes algorithm
2. **Early Neural Networks**
    - Toy forms as early as 1950s
    - Efficient leaning changed in 1980s by backpropagration algorithm using gradient-descent optimization
    - First application LeNet by Yann LeCun, recognizing ZIP code 
3. **Kernel Methods**
    - Kernel methods are group of classification algorithms (prefer by 2010)
        - Support Vector Machine (SVM)
            - Goal: finding good decision boundaries to split two categories
            - Two steps:
                - Map to high-dimensional representation where the decision boundary can expressed as a hyperplane
                - Maximizing the margin: A good decision boundary is computed by trying maximize the distance
            - Cons
                - Hard (computationally) in practice
                - *Kernel trick* to resolve it by kernel function (typically crafted by hand rather than learned from data)
                    - Map any two points to the distance between them in the new space, by passing the computation of new representation
                    - Then SVM only the separaton hyperplane is learned
                - Hard to scale to large dataset
4. **Decision Trees, Random Forests, and Gradient Boosting Machines**
    - Decision tree: flowchart-like strcture (2000s)
    - Random Forest: build a large number of decision trees and then ensembiling their outputs (*bagging*)
    - Gradient Boosting Machines: iteratively train new models that specialize in addressing the weak points of previous models, aka. gradient boosting method.
5. **Back to Neural Networks**, after 2010

### Deep Learning

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