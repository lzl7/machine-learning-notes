# 1. Cython
Cython is the superset of Python, most of the python could actually use as Cython.

Cython is not JIT.

The way it works is to convert the python code into C, compiple and run the C code; instead of passing it to Python interpreter. Hence, using Cython, it is actually writting C code with high level python syntax.

> Unlike Numba, all Cython code should be separated from regular Python code in special files.

Cython is widely used in pandas, scikit-learn scipy, Spacy etc.

## Tricks
- Load Cython in notebook: `%load_ext Cython`
- `%%cython` annotation on the function could fasten the python speed since it compiles the code
- Extension `cdef` defines the C data type of the return value and of each variable. Example: `cdef int a = 1`

# 2. Numba
> Numba is a just-in-time (JIT) compiler that analyzes bytecode and translates Python code to native machine instructions (via LLVM) both for CPU and GPU. The code can be compiled at import time, runtime, or ahead of time.

Numba uses LLVM as the backend.

## Modes
Two modes: `nopython` and `object`. `nopython` generates the native code directly while `object` use python object and python C API (perf might not have significant improvement).

## How to use
With decorator for the function: `@numba.jit(nopython=True)`

## Reference
- https://rushter.com/blog/numba-cython-python-optimization/
- https://medium.com/@hiromi_suenaga/machine-learning-1-lesson-7-69c50bc5e9af