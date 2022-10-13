"""Operator implementations."""

from array import array
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks
import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        
        # should be only 1 input
        x, _ = node.inputs
        return (out_grad ** (self.scalar - 1),)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a: NDArray, b: NDArray):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return out_grad * rhs ** (-1), out_grad * (-lhs / rhs ** 2)
        ### END YOUR SOLUTION

def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a: NDArray):
        ### BEGIN YOUR SOLUTION
        if not self.axes:
            return array_api.swapaxes(a, -1, -2)
        return array_api.swapaxes(a, *self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]
        return (transpose(out_grad, self.axes), )
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]
        return (reshape(out_grad, x.shape), )
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # source: https://forum.dlsyscourse.org/t/intuiton-and-math-behind-the-backward-summation-and-broadcast-operations/2068/4
        x = node.inputs[0]
        x_dim = len(x.shape)
        out_grad_dim = len(out_grad.shape)
        # aggregating axes to "reverse" broadcast
        agg_axes = []
        expand_dim_num = out_grad_dim - x_dim
        # boardcast first add axis from left
        agg_axes.extend([i for i in range(expand_dim_num)])
        # boardcast will replicate the axis with length 1
        agg_axes.extend([i + expand_dim_num for i in range(x_dim) if x.shape[i] == 1])
        agg_axes = tuple(agg_axes)
        return (reshape(summation(out_grad, agg_axes), x.shape), )
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # source: https://forum.dlsyscourse.org/t/intuiton-and-math-behind-the-backward-summation-and-broadcast-operations/2068
        x = node.inputs[0]
        grad_shape = list(out_grad.shape)
        if not self.axes:
            grad_shape = [1 for i in range(len(x.shape))]
        else:
            for axis in self.axes:
                grad_shape.insert(axis, 1)
        return (broadcast_to(reshape(out_grad, grad_shape), x.shape), )
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        dim_lhs = len(lhs.shape)
        dim_rhs = len(rhs.shape)
        dim_out_grad = len(out_grad.shape)
        if dim_lhs == dim_rhs:
            return out_grad @ transpose(rhs), transpose(lhs) @ out_grad
        else:
            # (6, 6, 5, 4) @ (4, 3) -> (6, 6, 5, 3)
            if dim_out_grad > dim_lhs:
                sum_axis = tuple(range(dim_out_grad - dim_lhs))
                return summation(out_grad @ transpose(rhs), sum_axis), transpose(lhs) @ out_grad
            else:
                sum_axis = tuple(range(dim_out_grad - dim_rhs))
                return out_grad @ transpose(rhs), summation(transpose(lhs) @  out_grad, sum_axis)

        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a: NDArray):
        ### BEGIN YOUR SOLUTION
        return array_api.negative(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (-out_grad, )
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a: NDArray):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]
        return (out_grad / x, )
        ### END YOUR SOLUTION

class Average(TensorOp):
    def __init__(self, axes = None):
        self.axes = None

    def compute(self, a: NDArray):
        return array_api.mean(a, self.axes)
    
    def gradient(self, out_grad, node):
        x = node.inputs[0]
        return (broadcast_to(out_grad, x.shape), )

def mean(a, axis = None):
    return Average(axis)(a)



def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]
        return (out_grad * exp(x),)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a: NDArray):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0.)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]
        return (out_grad * (x.realize_cached_data() > 0.), )
        ### END YOUR SOLUTION

def relu(a):
    return ReLU()(a)

