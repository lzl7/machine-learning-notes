# Broadcasting

N dimensional data usually known as rank n tensor. For example:  Scalar: rank 0 vector; Vector: rank 1 vector

**Broadcasting** usually means copy one of more axes of the tensor to allow it has the same shape as other tensor. In the underlying, it doesn't actually do the copy, but store kind of internal indicator that pretend it is rank n tensor. When accessing the data, rather than going to next row/next scalar, it goes back to where it came from.

*When it does broadcasting?*
When the size are different, for example `a>0`, `a+1`, `a*2` where a is a tensor.

When operating on two tensors/arrays, Pytorch/Numpy compares the shapes element-wise. It starts with the **trailing dimensions**. Two dimensions are compatible if:
- They are equal, or
- One of them is 1

## Reshaping

### Methods
- `np.expand_dimc(c,1)`, it means insert a length 1 axis/dimension there.
- Another easiest way is using special index `None`. It creates a new axis in that location of length 1. For example, c is array [1,2,3] (i.e. shape is (3,)):
  - `c[None].shape` is (1,3)
  - `c[:,None].shape` is (3,1)
  - `c[None,:,None].shape` is (1,3,1)
- `np.broadcast_to(c,(3,3))`, broad cast it to shape (3,3)
- `view` in Pytorch
  - If you don't know the number in certain dimension, use `-1`.
- `squeeze` in Pytorch
- `resize` in python

Pytorch also has the other functions help to reshape:
- `reshape`. The difference comparing with numpy.reshape is that it will always copy the memory while [numpy's](https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html) might not.

## Contigious vs non-contigious tensor
This is to indicate in the underlying the memory is continuously stored or not. You could check whether a tensor is contigious or not via method `is_contiguous()`. 

*Why this happens?* When creating the new tensor with sharing the same underlying data but different shapes. Like `transpose()`.

*How to ensure the tensor is `contiguous`?*
Use `contigous` method which would make a copy of tensor and the order of the elements would the same as the tensor shape as if it created from scratch.

Please note that, if the tensor already `contiguous`, it will just return itself.

Example:
```
t = torch.tensor(np.array([[0., 1.], [2., 3.], [4., 5.]]))
t1 = t.contiguous() 
id(t) == id(t1) # True
t2 = t.transpose(1,0) # transpose dim 1 and 0
t2.is_contiguous() # False
id(t) == id(t2) # False
id(t.data) == id(t2.data) # True
```

### Differences of the reshaping method
- `transpose()`: doesn't generate new tensor with new layout, instead it modifies the meta info in the Tensor object with the correct offset/stribe for the new shape. That is, it share the memory with the original data.
- `resize()` doesn't copy memory
- `reshape()` may copy memory if need
- `view()` doesn't copy memory

*`Transpose` vs. `view`*
`transpose` and `view` are used to change the shape of tensor, and the only difference is that `view` could only work on `contiguous` tensor while `transpose` can work on either.

*`Transpose` vs `Permute`*
`Transpose` can only works on swap two dimension while `Permute` could swap all dimensions.

## Size in pytorch
`Tensor.size()` and `Tensor.shape` are the same, but later is more friendly to numpy users.

## Data source
`Tensor.data` is a method you could access to the underlying data source. But `id(Tensor.data)` doesn't really give you the raw data source pointer info but the method `data_ptr()` does.

## Library
[TensorLy](https://github.com/tensorly/tensorly)

## Matrix mutipulation
[matrixmultiplication.xyz](http://matrixmultiplication.xyz/)

## Reference
- https://medium.com/@hiromi_suenaga/machine-learning-1-lesson-9-689bbc828fd2
- https://jdhao.github.io/2019/07/10/pytorch_view_reshape_transpose_permute/