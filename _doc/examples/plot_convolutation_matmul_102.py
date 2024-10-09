"""
==========================================
102: Convolution and Matrix Multiplication
==========================================

The `convolution <https://en.wikipedia.org/wiki/Kernel_(image_processing)>`_
is a well known image transformation used to transform an image.
It can be used to blur, to compute the gradient in one direction and
it is widely used in deep neural networks.
Having a fast implementation is important.

numpy
=====

Image have often 4 dimensions (N, C, H, W) = (batch, channels, height, width).
Let's first start with a 2D image.
"""

from typing import Sequence
import numpy as np
from numpy.testing import assert_almost_equal
from onnx.reference import ReferenceEvaluator
from onnx_array_api.light_api import start
from onnx_array_api.plotting.graphviz_helper import plot_dot
from onnxruntime import InferenceSession
from torch import from_numpy
from torch.nn import Fold, Unfold
from torch.nn.functional import conv_transpose2d, conv2d
from experimental_experiment.gradient.grad_helper import (
    onnx_derivative,
    DerivativeOptions,
)


shape = (5, 7)
N = np.prod(shape)
data = np.arange(N).astype(np.float32).reshape(shape)
# data[:, :] = 0
# data[2, 3] = 1
data.shape

#############################
# Let's a 2D kernel, the same one.

kernel = (np.arange(9) + 1).reshape(3, 3).astype(np.float32)
kernel


###############################
# raw convolution
# +++++++++++++++
#
# A raw version of a 2D convolution.


def raw_convolution(data: np.array, kernel: Sequence[int]) -> np.array:
    rx = (kernel.shape[0] - 1) // 2
    ry = (kernel.shape[1] - 1) // 2
    res = np.zeros(data.shape, dtype=data.dtype)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for x in range(kernel.shape[0]):
                for y in range(kernel.shape[1]):
                    a = i + x - rx
                    b = j + y - ry
                    if a < 0 or b < 0 or a >= data.shape[0] or b >= data.shape[1]:
                        continue
                    res[i, j] += kernel[x, y] * data[a, b]
    return res


res = raw_convolution(data, kernel)
res.shape

############################
# Full result.

res


###########################
# With pytorch
# ++++++++++++
#
# *pytorch* is optimized for deep learning and prefers 4D tenors
# to represent multiple images. We add two empty dimension
# to the previous example.


rest = conv2d(
    from_numpy(data[np.newaxis, np.newaxis, ...]),
    from_numpy(kernel[np.newaxis, np.newaxis, ...]),
    padding=(1, 1),
)
rest.shape

############################
# Full result.

rest

############################
# Everything works.


assert_almost_equal(res, rest[0, 0].numpy())

##################################
# using Gemm?
# +++++++++++
#
# A fast implementation could reuse whatever exists with a fast implementation
# such as a matrix multiplication. The goal is to transform the tensor `data`
# into a new matrix which can be mutiplied with a flatten kernel and finally
# reshaped into the expected result. pytorch calls this function
# `Unfold <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html>`_.
# This function is also called
# `im2col <https://caffe.berkeleyvision.org/tutorial/layers/im2col.html>`_.


unfold = Unfold(kernel_size=(3, 3), padding=(1, 1))(
    from_numpy(data[np.newaxis, np.newaxis, ...])
)
unfold.shape

#########################################
# We then multiply this matrix with the flattened kernel and reshape it.


impl = kernel.flatten() @ unfold.numpy()
impl = impl.reshape(data.shape)
impl.shape

##########################
# Full result.

impl

#########################
# Everything works as expected.


assert_almost_equal(res, impl)


##############################
# What is ConvTranspose?
# ++++++++++++++++++++++
#
# Deep neural network are trained with a stochastic gradient descent.
# The gradient of every layer needs to be computed including the gradient
# of a convolution transpose. That seems easier with the second expression
# of a convolution relying on a matrix multiplication and function `im2col`.
# `im2col` is just a new matrix built from `data` where every value was
# copied in 9=3x3 locations. The gradient against an input value `data[i,j]`
# is the sum of 9=3x3 values from the output gradient. If `im2col` plays
# with indices, the gradient requires to do the same thing in the other way.


# impl[:, :] = 0
# impl[2, 3] = 1
impl

################################
# ConvTranspose...


ct = conv_transpose2d(
    from_numpy(impl.reshape(data.shape)[np.newaxis, np.newaxis, ...]),
    from_numpy(kernel[np.newaxis, np.newaxis, ...]),
    padding=(1, 1),
).numpy()
ct

##############################
# And now the version with `col2im` or
# `Fold <https://pytorch.org/docs/stable/generated/torch.nn.Fold.html#torch.nn.Fold>`_
# applied on the result product of the output from `Conv` and the kernel:
# the output of `Conv` is multiplied by every coefficient of the kernel.
# Then all these matrices are concatenated to build a matrix of the same
# shape of `unfold`.


p = kernel.flatten().reshape((-1, 1)) @ impl.flatten().reshape((1, -1))
p.shape

#############################
# Fold...


fold = Fold(kernel_size=(3, 3), output_size=(5, 7), padding=(1, 1))(
    from_numpy(p[np.newaxis, ...])
)
fold.shape

########################
# Full result.

fold

##############################
# onnxruntime-training
# ====================
#
# Following lines shows how :epkg:`onnxruntime` handles the
# gradient computation. This section still needs work.
#
# Conv
# ++++


model = (
    start(ir_version=9, opset=18)
    .vin("X", shape=[1, 1, None, None])
    .cst(kernel[np.newaxis, np.newaxis, ...])
    .rename("W")
    .bring("X", "W")
    .Conv(pads=[1, 1, 1, 1])
    .rename("Y")
    .vout()
    .to_onnx()
)
plot_dot(model)


################################
# Execution


ref = ReferenceEvaluator(model)
ref.run(None, {"X": data[np.newaxis, np.newaxis, ...]})[0]


##############################
# Gradient


grad = onnx_derivative(
    model, options=DerivativeOptions.FillGrad | DerivativeOptions.KeepOutputs, verbose=1
)
plot_dot(grad)


###############################
# Execution.


sess = InferenceSession(grad.SerializeToString(), providers=["CPUExecutionProvider"])
res = sess.run(
    None,
    {
        "X": data[np.newaxis, np.newaxis, ...],
        "W": kernel[np.newaxis, np.newaxis, ...],
    },
)
res

################################
# ConvTranspose
# +++++++++++++


model = (
    start(ir_version=9, opset=18)
    .vin("X", shape=[1, 1, None, None])
    .cst(kernel[np.newaxis, np.newaxis, ...])
    .rename("W")
    .bring("X", "W")
    .ConvTranspose(pads=[1, 1, 1, 1])
    .rename("Y")
    .vout()
    .to_onnx()
)
plot_dot(model)

############################
# Execution.

sess = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
ct = sess.run(None, {"X": impl[np.newaxis, np.newaxis, ...]})[0]
ct


###############################
# im2col and col2im
# =================
#
# Function `im2col` transforms an image so that the convolution of this image
# can be expressed as a matrix multiplication. It takes the image and the kernel shape.


def _get_indices(i: int, shape: Sequence[int]) -> np.array:
    res = np.empty((len(shape),), dtype=np.int64)
    k = len(shape) - 1
    while k > 0:
        m = i % shape[k]
        res[k] = m
        i -= m
        i /= shape[k]
        k -= 1
    res[0] = i
    return res


def _is_out(ind: Sequence[int], shape: Sequence[int]) -> bool:
    for i, s in zip(ind, shape):
        if i < 0:
            return True
        if i >= s:
            return True
    return False


def im2col_naive_implementation(
    data: np.array, kernel_shape: Sequence[int], fill_value: int = 0
) -> np.array:
    """
    Naive implementation for `im2col` or
    :func:`torch.nn.Unfold` (but with `padding=1`).

    :param image: image (float)
    :param kernel_shape: kernel shape
    :param fill_value: fill value
    :return: result
    """
    if not isinstance(kernel_shape, tuple):
        raise TypeError(f"Unexpected type {type(kernel_shape)!r} for kernel_shape.")
    if len(data.shape) != len(kernel_shape):
        raise ValueError(f"Shape mismatch {data.shape!r} and {kernel_shape!r}.")
    output_shape = data.shape + kernel_shape
    res = np.empty(output_shape, dtype=data.dtype)
    middle = np.array([-m / 2 for m in kernel_shape], dtype=np.int64)
    kernel_size = np.prod(kernel_shape)
    data_size = np.prod(data.shape)
    for i in range(data_size):
        for j in range(kernel_size):
            i_data = _get_indices(i, data.shape)
            i_kernel = _get_indices(j, kernel_shape)
            ind = i_data + i_kernel + middle
            t_data = tuple(i_data)
            t_kernel = tuple(i_kernel)
            i_out = t_data + t_kernel
            res[i_out] = fill_value if _is_out(ind, data.shape) else data[tuple(ind)]
    return res


v = np.arange(5).astype(np.float32)
w = im2col_naive_implementation(v, (3,))
w

##################################
# All is left is the matrix multiplication.


k = np.array([1, 1, 1], dtype=np.float32)
conv = w @ k
conv

######################################
# Let's compare with the numpy function.


np.convolve(v, k, mode="same")


#########################################################
# ..math::
#
#     conv(v, k) = im2col(v, shape(k)) \; k = w \; k` where `w = im2col(v, shape(k))
#
# In deep neural network, the gradient is propagated from the last layer
# to the first one. At some point, the backpropagation produces the gradient
# :math:`\frac{d(E)}{d(conv)}`, the gradient of the error against
# the outputs of the convolution layer. Then
# :math:`\frac{d(E)}{d(v)} = \frac{d(E)}{d(conv(v, k))}\frac{d(conv(v, k))}{d(v)}`.
#
# We need to compute
# :math:`\frac{d(conv(v, k))}{d(v)} = \frac{d(conv(v, k))}{d(w)}\frac{d(w)}{d(v)}`.
#
# We can say that :math:`\frac{d(conv(v, k))}{d(w)} = k`.
#
# That leaves :math:`\frac{d(w)}{d(v)} = \frac{d(im2col(v, shape(k)))}{d(v)}`.
# And this last term is equal to :math:`im2col(m, shape(k))` where :math:`m`
# is a matrix identical to :math:`v` except that all not null parameter
# are replaced by 1. To summarize:
# :math:`\frac{d(im2col(v, shape(k)))}{d(v)} = im2col(v \neq 0, shape(k))`.
#
# Finally:
#
# .. math::
#
#   \frac{d(E)}{d(v)} = \frac{d(E)}{d(conv(v, k))}\frac{d(conv(v, k))}{d(v)} =
#   \frac{d(E)}{d(conv(v, k))} \; k \; im2col(v \neq 0, shape(k))
#
# Now, :math:`im2col(v \neq 0, shape(k))` is a very simple matrix with only ones or zeros.
# Is there a way we can avoid doing the matrix multiplication but simply
# adding terms? That's the purpose of function ``col2im`` defined so that:
#
# .. math::
#
#   \frac{d(E)}{d(v)} = \frac{d(E)}{d(conv(v, k))} \; k \; i
#   m2col(v \neq 0, shape(k)) = col2im\left(\frac{d(E)}{d(conv(v, k))} \; k, shape(k) \right)
