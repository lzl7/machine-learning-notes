# Broadcasting

N dimentional data usually known as rank n tensor. For example:  Scalar: rank 0 vector; Vector: rank 1 vector

**Broadcasting** usually means copy one of more axes of the tensor to allow it has the same shape as other tensor. In the underlying, it doesn't actually do the copy, but store kind of internal indicator that pretend it is rank n tensor. When accessing the data, rather than going to next row/next scalar, it goes back to where it came from.

*When it does broadcasting?*
When the size are different, for example `a>0`, `a+1`, `a*2` where a is a tensor.

When operating on two tensors/arrays, Pytorch/Numpy compares the shapes element-wise. It starts with the **trailing dimensions**. Two dimensions are compatible if:
- They are equal, or
- One of them is 1

## Reshaping
- `np.expand_dimc(c,1)`, it means insert a length 1 axis/dimension there.
- Another easiest way is using special index `None`. It creates a new axis in that location of length 1. For example, c is array [1,2,3] (i.e. shape is (3,)):
  - `c[None].shape` is (1,3)
  - `c[:,None].shape` is (3,1)
  - `c[None,:,None].shape` is (1,3,1)
- `np.broadcast_to(c,(3,3))`, broad cast it to shape (3,3)
- `view` in Pytorch
- `sequeeze` in Pytorch

Pytorch also has the other functions help to reshape:
- `reshape`

## Library
[TensorLy](https://github.com/tensorly/tensorly)

## Matrix mutipulation
[matrixmultiplication.xyz](http://matrixmultiplication.xyz/)

## Reference
- https://medium.com/@hiromi_suenaga/machine-learning-1-lesson-9-689bbc828fd2