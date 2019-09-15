# Softmax vs. Sigmoid
> Softmax is specifically the activation function we use for categorical predictions where we only ever want to predict one of those things

> If you are doing multi-label prediction, rather than using softmax, instead use sigmoid.

# Othes popular activation functions
- ReLU, `max(x,0)` (popular ~3 years ago)
- Leaky ReLU, `max(0.1x,x)` (1 years ago)
- ELU, `x if x>=0, else α(e^x-1)` (1 year)

The choice of activation function doesn’t matter terribly much actually. 

## Reference
- https://medium.com/@hiromi_suenaga/machine-learning-1-lesson-9-689bbc828fd2