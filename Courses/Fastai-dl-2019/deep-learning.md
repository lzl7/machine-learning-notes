# What is deep leanring

An interesting point of view from FastAI is: Deep learning is a class of algorithms that have following properties:
- Infinitely flexible function: Neural Network
- All purpose parameter fitting: Gradient Descent
- Fast and scalable: GPU

> A combination of linear layer followed by an element-wise nonlinear function allows us to create arbitrarily complex shapes — this is the essence of the universal approximation theorem.

There are several important parameters need to set for deep leanring:
- Learning rate - how fast to update the weights (parameters)
    - Too large: diverge instead of converge
    - Too small: slow converge or not converge
- Epoch
  - Make sure it not "overfitting" but converge enough

## Reference
- https://medium.com/@hiromi_suenaga/deep-learning-2-part-1-lesson-1-602f73869197