# Useful methods in pytroch
- `.cudoa()`, move the tensor/data to GPU

# Defining Module
When creating a module which inherits from `nn.Module` directly/indirectly, always call `super().__init__()` in the constructor first.

Note: when calling the module like as a function, it actually call the the method `forward`

With `variable`, we could get free automatic differention, Pytorch can automatically differentiate pretty much any tensor. To do automatic differentiation, it has to keep track of exactly how something was calculated. In Pytorch, it works byby wrapping a tensor in Variable.

## Training loop
To get gradient, we call the function `loss.backward()`, where the `loss` is the object that contains the loss. What `backward` does is calculate the gradients and stores them inside. Basically, for each of the weights/parameters that used to calculate the gradient are stored in `.grad`

`optimizer.step()` is to make an update to its parameters.

`optimizer.zero_grad` is to clear (i.e. reset to zero) all the gradients for the variables.

The training loop step:
- Take one/more mini batch of data from data loader
- Calculate prediction with the network
- Calculate loss from the prediction and actuals
- zero the gradients
- Calculate the gradients via `backward`
- Update the weights via `step`

*Why need multiple epochs?*
It might not get very far if the learning rate is tiny. *why not set learning rate large enough?* It is hard to do it, if the lr is too large it will also get into trouble.

# Data Loader
Dataset is basically something looks like a list, has the length and ability to index into it like a list.

DataLoader takes a dataset and then could be iterable. You could specify whether shuffle or not, what is the batch size.

# Learning Rate
## Learning rate annealing/decay

# Regularization
A really simple common approach to regularization in all of machine learning is something called L2 regularization. L1 is the absolute value of the weights average. L2 is the squares of the weights themselves.

> Regularization in neural nets means either weight decay (also known as “kind of” L2 regularization) or dropout.

There are several ways to do the regularization:
- L2 regularizaiton: Modify loss function to add the square penalty
- Weight decay

Weight decay happens after each batch. L2 regularization and weight decay are mathmatically identical.

*With regularization, the training loss is worse, is that expected?*

Yes, because you are penalizing it. But the loss in validation set is better since the penaty on training set makes it generalize better. That is the final number get from the training set shouldn't be better, but it can train sometimes more quickly.

Weight decay doesn't always make hte function surface smoother, but it may help if you are having trouble training a function.

## Reference
- https://medium.com/@hiromi_suenaga/machine-learning-1-lesson-9-689bbc828fd2