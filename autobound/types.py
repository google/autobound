# Copyright 2023 The autobound Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Core types used in this package."""

import abc
from typing import NewType, Union
import typing_extensions


# An interval is represented as a pair (a, b) of NDArrays of the same shape.
# It represents the set of NDArrays {x: a <= x <= b}, where the inequalities
# are elementwise.
Interval = tuple['NDArray', 'NDArray']


# A polynomial whose coefficients can be either NDArrays or Intervals.  The
# mathematical formula for the polynomial can be different for different types
# of polynomials (see below).
IntervalPolynomial = tuple[Union['NDArray', Interval], ...]


# A degree d Taylor enclosure is a polynomial, represented as a tuple of
# coefficients (c0, c1, ..., c_d).  Each coefficient can be either an NDArray
# or an Interval.  In the case where all coefficients are scalar, the
# corresponding polynomial is:
#
#     sum_{i=0}^d c_i (x - x_0)^i.
#
# In the case where x is a rank-k tensor, (x - x_0)^i is replaced by a repeated
# outer product that yields a tensor of rank k*i, and c_i (x - x_0)^i is
# replaced by an inner product that contracts over the last k*i axes of c_i.
#
# Written as Numpy code, the ith term of the sum becomes:
#
#     np.tensordot(c_i, outer_power(x-x_0, i), k*i)
#
# where:
#
#     outer_power(x-x_0, 0) = 1
#     outer_power(x-x_0, i) = np.tensordot(outer_power(x-x_0, i-1), x-x_0, 0).
#
# A given tuple defines a valid Taylor enclosure of a function f at a point
# x0 if f(x) belongs to the interval obtained by evaluating the polynomial
# at x.
#
# Typically the last coefficient of a degree d Taylor enclosure will be an
# Interval, and the first d coefficients will be NDArrays (but this is not
# required).
TaylorEnclosure = NewType('TaylorEnclosure', IntervalPolynomial)


# A degree d _elementwise_ Taylor enclosure of a function f at a point x0
# is a tuple (c_0, c_1, ... c_d), where each coefficient c_i is either an
# NDArray or an Interval, such that
#
#   f(x) in sum_{i=0}^d c_i * (x - x_0)**i for all x.
#
# In general only the last coefficient is an Interval, and the first d-1
# coefficients are NDArrays that match the coefficients of the degree d-1
# Taylor series expansion of f at x0.
ElementwiseTaylorEnclosure = NewType('ElementwiseTaylorEnclosure',
                                     IntervalPolynomial)


# Type for objects that are accepted in place of NDArray arguments.
#
# Sequence[float] is not included in the Union, because for example
# jnp.add([1.], [2.]) does not work.
NDArrayLike = Union['NDArray', int, float]

# Similar types for Intervals, etc.
IntervalLike = tuple[NDArrayLike, NDArrayLike]
IntervalPolynomialLike = tuple[Union[NDArrayLike, IntervalLike], ...]
TaylorEnclosureLike = IntervalPolynomialLike
ElementwiseTaylorEnclosureLike = IntervalPolynomialLike


# pylint: disable=multiple-statements


@typing_extensions.runtime_checkable
class NDArray(typing_extensions.Protocol):
  """Protocol for numpy-like n-dimensional arrays.

  This class is used in type annotations when writing functions that
  accept numpy, jax.numpy, or tensorflow.experimental.numpy ndarrays.

  Example usage:

    def mean_squared_error(a: NDArray, x: NDArray, b: NDArray) -> float:
      return float(((a @ x - b)**2).mean())
  """

  #
  # Python magic methods.
  #
  @abc.abstractmethod
  def __abs__(self) -> 'NDArray': pass
  @abc.abstractmethod
  def __add__(self, o: 'NDArrayLike') -> 'NDArray': pass
  @abc.abstractmethod
  def __and__(self, o: 'NDArrayLike') -> 'NDArray': pass
  @abc.abstractmethod
  def __bool__(self) -> 'NDArray': pass
  @abc.abstractmethod
  def __complex__(self) -> complex: pass
  @abc.abstractmethod
  def __eq__(self, o: 'NDArrayLike') -> 'NDArray': pass
  @abc.abstractmethod
  def __float__(self) -> float: pass
  @abc.abstractmethod
  def __floordiv__(self, o: 'NDArrayLike') -> 'NDArray': pass
  @abc.abstractmethod
  def __ge__(self, o: 'NDArrayLike') -> 'NDArray': pass
  @abc.abstractmethod
  def __getitem__(self, key) -> 'NDArray': pass
  @abc.abstractmethod
  def __gt__(self, o: 'NDArrayLike') -> 'NDArray': pass
  @abc.abstractmethod
  def __index__(self) -> int: pass
  @abc.abstractmethod
  def __int__(self) -> int: pass
  @abc.abstractmethod
  def __invert__(self) -> 'NDArray': pass
  @abc.abstractmethod
  def __iter__(self): pass
  @abc.abstractmethod
  def __le__(self, o: 'NDArrayLike') -> 'NDArray': pass
  @abc.abstractmethod
  def __len__(self) -> int: pass
  @abc.abstractmethod
  def __lt__(self, o: 'NDArrayLike') -> 'NDArray': pass
  @abc.abstractmethod
  def __matmul__(self, o: 'NDArrayLike') -> 'NDArray': pass
  @abc.abstractmethod
  def __mod__(self, o: 'NDArrayLike') -> 'NDArray': pass
  @abc.abstractmethod
  def __mul__(self, o: 'NDArrayLike') -> 'NDArray': pass
  @abc.abstractmethod
  def __ne__(self, o: 'NDArrayLike') -> 'NDArray': pass
  @abc.abstractmethod
  def __neg__(self) -> 'NDArray': pass
  @abc.abstractmethod
  def __or__(self, o: 'NDArrayLike') -> 'NDArray': pass
  @abc.abstractmethod
  def __pos__(self) -> 'NDArray': pass
  @abc.abstractmethod
  def __pow__(self, o: 'NDArrayLike') -> 'NDArray': pass
  @abc.abstractmethod
  def __radd__(self, o: 'NDArrayLike') -> 'NDArray': pass
  @abc.abstractmethod
  def __rand__(self, o: 'NDArrayLike') -> 'NDArray': pass
  @abc.abstractmethod
  def __rfloordiv__(self, o: 'NDArrayLike') -> 'NDArray': pass
  @abc.abstractmethod
  def __rmatmul__(self, o: 'NDArrayLike') -> 'NDArray': pass
  @abc.abstractmethod
  def __rmod__(self, o: 'NDArrayLike') -> 'NDArray': pass
  @abc.abstractmethod
  def __rmul__(self, o: 'NDArrayLike') -> 'NDArray': pass
  @abc.abstractmethod
  def __ror__(self, o: 'NDArrayLike') -> 'NDArray': pass
  @abc.abstractmethod
  def __rpow__(self, o: 'NDArrayLike') -> 'NDArray': pass
  @abc.abstractmethod
  def __rsub__(self, o: 'NDArrayLike') -> 'NDArray': pass
  @abc.abstractmethod
  def __rtruediv__(self, o: 'NDArrayLike') -> 'NDArray': pass
  @abc.abstractmethod
  def __rxor__(self, o: 'NDArrayLike') -> 'NDArray': pass
  @abc.abstractmethod
  def __sub__(self, o: 'NDArrayLike') -> 'NDArray': pass
  @abc.abstractmethod
  def __truediv__(self, o: 'NDArrayLike') -> 'NDArray': pass
  @abc.abstractmethod
  def __xor__(self, o: 'NDArrayLike') -> 'NDArray': pass

  #
  # Properties.
  #
  @property
  def dtype(self): raise NotImplementedError(self.__class__)
  @property
  def ndim(self) -> int: raise NotImplementedError(self.__class__)
  @property
  def shape(self): raise NotImplementedError(self.__class__)
  @property
  def size(self): raise NotImplementedError(self.__class__)
  @property
  def T(self): raise NotImplementedError(self.__class__)

  #
  # Public methods.
  #
  @abc.abstractmethod
  def astype(self, *args, **kwargs) -> 'NDArray': pass
  @abc.abstractmethod
  def clip(self, *args, **kwargs) -> 'NDArray': pass
  @abc.abstractmethod
  def max(self, *args, **kwargs) -> 'NDArray': pass
  @abc.abstractmethod
  def mean(self, *args, **kwargs) -> 'NDArray': pass
  @abc.abstractmethod
  def min(self, *args, **kwargs) -> 'NDArray': pass
  @abc.abstractmethod
  def ravel(self, *args) -> 'NDArray': pass
  @abc.abstractmethod
  def reshape(self, *args, **kwargs) -> 'NDArray': pass
  # TODO(mstreeter): list sum() here, once tnp supports it.
  @abc.abstractmethod
  def tolist(self) -> 'NDArray': pass
  @abc.abstractmethod
  def transpose(self, *axes) -> 'NDArray': pass


@typing_extensions.runtime_checkable
class NumpyLike(typing_extensions.Protocol):
  """Numpy-like back end for performing basic operations on n-dim arrays.

  This class is used in type annotations when writing functions that work
  with an arbitrary Numpy-like backend.
  """

  @abc.abstractmethod
  def abs(self, a: NDArrayLike) -> NDArray: pass
  @abc.abstractmethod
  def absolute(self, a: NDArrayLike, *args, **kwargs) -> NDArray: pass
  @abc.abstractmethod
  def add(self, a: NDArrayLike, b: NDArrayLike) -> NDArray: pass
  @abc.abstractmethod
  def all(self, a: NDArrayLike, *args, **kwargs) -> NDArray: pass
  @abc.abstractmethod
  def allclose(self, *args, **kwargs): pass
  @abc.abstractmethod
  def amax(self, *args, **kwargs): pass
  @abc.abstractmethod
  def amin(self, *args, **kwargs): pass
  @abc.abstractmethod
  def angle(self, *args, **kwargs): pass
  @abc.abstractmethod
  def any(self, *args, **kwargs): pass
  @abc.abstractmethod
  def append(self, *args, **kwargs): pass
  @abc.abstractmethod
  def arange(self, *args, **kwargs): pass
  @abc.abstractmethod
  def arccos(self, *args, **kwargs): pass
  @abc.abstractmethod
  def arccosh(self, *args, **kwargs): pass
  @abc.abstractmethod
  def arcsin(self, *args, **kwargs): pass
  @abc.abstractmethod
  def arcsinh(self, *args, **kwargs): pass
  @abc.abstractmethod
  def arctan(self, *args, **kwargs): pass
  @abc.abstractmethod
  def arctan2(self, *args, **kwargs): pass
  @abc.abstractmethod
  def arctanh(self, *args, **kwargs): pass
  @abc.abstractmethod
  def argmax(self, *args, **kwargs): pass
  @abc.abstractmethod
  def argmin(self, *args, **kwargs): pass
  @abc.abstractmethod
  def argsort(self, *args, **kwargs): pass
  @abc.abstractmethod
  def around(self, *args, **kwargs): pass
  @abc.abstractmethod
  def array(self, obj, **kwargs) -> NDArray: pass
  @abc.abstractmethod
  def array_equal(self, *args, **kwargs): pass
  @abc.abstractmethod
  def asarray(self, a) -> NDArray: pass
  @abc.abstractmethod
  def atleast_1d(self, *args, **kwargs): pass
  @abc.abstractmethod
  def atleast_2d(self, *args, **kwargs): pass
  @abc.abstractmethod
  def atleast_3d(self, *args, **kwargs): pass
  @abc.abstractmethod
  def average(self, *args, **kwargs): pass
  @abc.abstractmethod
  def bitwise_and(self, *args, **kwargs): pass
  @abc.abstractmethod
  def bitwise_not(self, *args, **kwargs): pass
  @abc.abstractmethod
  def bitwise_or(self, *args, **kwargs): pass
  @abc.abstractmethod
  def bitwise_xor(self, *args, **kwargs): pass
  @abc.abstractmethod
  def bool_(self, *args, **kwargs): pass
  @abc.abstractmethod
  def broadcast_arrays(self, *args) -> list[NDArray]: pass
  @abc.abstractmethod
  def broadcast_to(self, *args, **kwargs): pass
  @abc.abstractmethod
  def cbrt(self, *args, **kwargs): pass
  @abc.abstractmethod
  def ceil(self, *args, **kwargs): pass
  @abc.abstractmethod
  def clip(self, *args, **kwargs): pass
  @abc.abstractmethod
  def complex128(self, *args, **kwargs): pass
  @abc.abstractmethod
  def complex64(self, *args, **kwargs): pass
  @abc.abstractmethod
  def complex_(self, *args, **kwargs): pass
  @abc.abstractmethod
  def compress(self, *args, **kwargs): pass
  @abc.abstractmethod
  def concatenate(self, *args, **kwargs): pass
  @abc.abstractmethod
  def conj(self, *args, **kwargs): pass
  @abc.abstractmethod
  def conjugate(self, *args, **kwargs): pass
  @abc.abstractmethod
  def copy(self, *args, **kwargs): pass
  @abc.abstractmethod
  def cos(self, *args, **kwargs): pass
  @abc.abstractmethod
  def cosh(self, *args, **kwargs): pass
  @abc.abstractmethod
  def count_nonzero(self, *args, **kwargs): pass
  @abc.abstractmethod
  def cross(self, *args, **kwargs): pass
  @abc.abstractmethod
  def cumprod(self, *args, **kwargs): pass
  @abc.abstractmethod
  def cumsum(self, *args, **kwargs): pass
  @abc.abstractmethod
  def deg2rad(self, *args, **kwargs): pass
  @abc.abstractmethod
  def diag(self, *args, **kwargs) -> NDArray: pass
  @abc.abstractmethod
  def diag_indices(self, *args, **kwargs): pass
  @abc.abstractmethod
  def diagflat(self, *args, **kwargs): pass
  @abc.abstractmethod
  def diagonal(self, *args, **kwargs): pass
  @abc.abstractmethod
  def diff(self, *args, **kwargs): pass
  @abc.abstractmethod
  def divide(self, a: NDArrayLike, b: NDArrayLike) -> NDArray: pass
  @abc.abstractmethod
  def divmod(self, *args, **kwargs): pass
  @abc.abstractmethod
  def dot(self, *args, **kwargs): pass
  @abc.abstractmethod
  def dsplit(self, *args, **kwargs): pass
  @abc.abstractmethod
  def dstack(self, *args, **kwargs): pass
  @abc.abstractmethod
  def einsum(self, *args, **kwargs): pass
  @abc.abstractmethod
  def empty(self, *args, **kwargs): pass
  @abc.abstractmethod
  def empty_like(self, *args, **kwargs): pass
  @abc.abstractmethod
  def equal(self, x1: NDArrayLike, x2: NDArrayLike, **kwargs) -> NDArray: pass
  @abc.abstractmethod
  def exp(self, x: NDArrayLike) -> NDArray: pass
  @abc.abstractmethod
  def exp2(self, *args, **kwargs): pass
  @abc.abstractmethod
  def expand_dims(self, *args, **kwargs): pass
  @abc.abstractmethod
  def expm1(self, *args, **kwargs): pass
  @abc.abstractmethod
  def eye(self, n: int, *args, **kwargs) -> NDArray: pass
  @abc.abstractmethod
  def fabs(self, *args, **kwargs): pass
  @abc.abstractmethod
  def finfo(self, *args, **kwargs): pass
  @abc.abstractmethod
  def fix(self, *args, **kwargs): pass
  @abc.abstractmethod
  def flip(self, *args, **kwargs): pass
  @abc.abstractmethod
  def fliplr(self, *args, **kwargs): pass
  @abc.abstractmethod
  def flipud(self, *args, **kwargs): pass
  @abc.abstractmethod
  def float16(self, *args, **kwargs): pass
  @abc.abstractmethod
  def float32(self, *args, **kwargs): pass
  @abc.abstractmethod
  def float64(self, *args, **kwargs): pass
  @abc.abstractmethod
  def float_(self, *args, **kwargs): pass
  @abc.abstractmethod
  def float_power(self, *args, **kwargs): pass
  @abc.abstractmethod
  def floor(self, *args, **kwargs): pass
  @abc.abstractmethod
  def floor_divide(self, *args, **kwargs): pass
  @abc.abstractmethod
  def full(self, *args, **kwargs): pass
  @abc.abstractmethod
  def full_like(self, *args, **kwargs): pass
  @abc.abstractmethod
  def gcd(self, *args, **kwargs): pass
  @abc.abstractmethod
  def geomspace(self, *args, **kwargs): pass
  @abc.abstractmethod
  def greater(self, x1: NDArrayLike, x2: NDArrayLike, **kwargs) -> NDArray:
    pass
  @abc.abstractmethod
  def greater_equal(self, x1: NDArrayLike, x2: NDArrayLike,
                    **kwargs) -> NDArray: pass
  @abc.abstractmethod
  def heaviside(self, *args, **kwargs): pass
  @abc.abstractmethod
  def hsplit(self, *args, **kwargs): pass
  @abc.abstractmethod
  def hstack(self, *args, **kwargs): pass
  @abc.abstractmethod
  def hypot(self, *args, **kwargs): pass
  @abc.abstractmethod
  def identity(self, *args, **kwargs): pass
  @abc.abstractmethod
  def iinfo(self, *args, **kwargs): pass
  @abc.abstractmethod
  def imag(self, *args, **kwargs): pass
  @abc.abstractmethod
  def inexact(self, *args, **kwargs): pass
  @abc.abstractmethod
  def inner(self, *args, **kwargs): pass
  @abc.abstractmethod
  def int16(self, *args, **kwargs): pass
  @abc.abstractmethod
  def int32(self, *args, **kwargs): pass
  @abc.abstractmethod
  def int64(self, *args, **kwargs): pass
  @abc.abstractmethod
  def int8(self, *args, **kwargs): pass
  @abc.abstractmethod
  def int_(self, *args, **kwargs): pass
  @abc.abstractmethod
  def isclose(self, *args, **kwargs): pass
  @abc.abstractmethod
  def iscomplex(self, *args, **kwargs): pass
  @abc.abstractmethod
  def iscomplexobj(self, *args, **kwargs): pass
  @abc.abstractmethod
  def isfinite(self, *args, **kwargs): pass
  @abc.abstractmethod
  def isinf(self, *args, **kwargs): pass
  @abc.abstractmethod
  def isnan(self, *args, **kwargs): pass
  @abc.abstractmethod
  def isneginf(self, *args, **kwargs): pass
  @abc.abstractmethod
  def isposinf(self, *args, **kwargs): pass
  @abc.abstractmethod
  def isreal(self, *args, **kwargs): pass
  @abc.abstractmethod
  def isrealobj(self, *args, **kwargs): pass
  @abc.abstractmethod
  def isscalar(self, *args, **kwargs): pass
  @abc.abstractmethod
  def issubdtype(self, *args, **kwargs): pass
  @abc.abstractmethod
  def ix_(self, *args, **kwargs): pass
  @abc.abstractmethod
  def kron(self, *args, **kwargs): pass
  @abc.abstractmethod
  def lcm(self, *args, **kwargs): pass
  @abc.abstractmethod
  def less(self, x1: NDArrayLike, x2: NDArrayLike, **kwargs) -> NDArray: pass
  @abc.abstractmethod
  def less_equal(self, x1: NDArrayLike, x2: NDArrayLike, **kwargs) -> NDArray:
    pass
  @abc.abstractmethod
  def linspace(self, *args, **kwargs): pass
  @abc.abstractmethod
  def log(self, x: NDArrayLike) -> NDArray: pass
  @abc.abstractmethod
  def log1p(self, x: NDArrayLike) -> NDArray: pass
  def log10(self, *args, **kwargs): pass
  @abc.abstractmethod
  def log2(self, *args, **kwargs): pass
  @abc.abstractmethod
  def logaddexp(self, *args, **kwargs): pass
  @abc.abstractmethod
  def logaddexp2(self, *args, **kwargs): pass
  @abc.abstractmethod
  def logical_and(self, a: NDArrayLike, b: NDArrayLike) -> NDArray: pass
  @abc.abstractmethod
  def logical_not(self, *args, **kwargs): pass
  @abc.abstractmethod
  def logical_or(self, *args, **kwargs): pass
  @abc.abstractmethod
  def logical_xor(self, *args, **kwargs): pass
  @abc.abstractmethod
  def logspace(self, *args, **kwargs): pass
  @abc.abstractmethod
  def matmul(self, *args, **kwargs): pass
  @abc.abstractmethod
  def max(self, *args, **kwargs): pass
  @abc.abstractmethod
  def maximum(self, a: NDArrayLike, b: NDArrayLike) -> NDArray: pass
  @abc.abstractmethod
  def mean(self, *args, **kwargs): pass
  @abc.abstractmethod
  def meshgrid(self, *args, **kwargs): pass
  @abc.abstractmethod
  def min(self, *args, **kwargs): pass
  @abc.abstractmethod
  def minimum(self, a: NDArrayLike, b: NDArrayLike) -> NDArray: pass
  @abc.abstractmethod
  def mod(self, *args, **kwargs): pass
  @abc.abstractmethod
  def moveaxis(self, *args, **kwargs): pass
  @abc.abstractmethod
  def multiply(self, a: NDArrayLike, b: NDArrayLike) -> NDArray: pass
  @abc.abstractmethod
  def nanmean(self, *args, **kwargs): pass
  @abc.abstractmethod
  def nanprod(self, *args, **kwargs): pass
  @abc.abstractmethod
  def nansum(self, *args, **kwargs): pass
  @abc.abstractmethod
  def ndarray(self, *args, **kwargs): pass
  @abc.abstractmethod
  def ndim(self, a: NDArrayLike) -> int: pass
  @abc.abstractmethod
  def negative(self, a: NDArrayLike) -> NDArray: pass
  @abc.abstractmethod
  def nextafter(self, *args, **kwargs): pass
  @abc.abstractmethod
  def nonzero(self, *args, **kwargs): pass
  @abc.abstractmethod
  def not_equal(self, *args, **kwargs): pass
  @abc.abstractmethod
  def object_(self, *args, **kwargs): pass
  @abc.abstractmethod
  def ones(self, shape, **kwargs) -> NDArray: pass
  @abc.abstractmethod
  def ones_like(self, a) -> NDArray: pass
  @abc.abstractmethod
  def outer(self, *args, **kwargs): pass
  @abc.abstractmethod
  def pad(self, *args, **kwargs): pass
  @abc.abstractmethod
  def polyval(self, *args, **kwargs): pass
  @abc.abstractmethod
  def positive(self, *args, **kwargs): pass
  @abc.abstractmethod
  def power(self, a, exponent) -> NDArray: pass
  @abc.abstractmethod
  def prod(self, *args, **kwargs): pass
  @abc.abstractmethod
  def promote_types(self, *args, **kwargs): pass
  @abc.abstractmethod
  def ptp(self, *args, **kwargs): pass
  @abc.abstractmethod
  def rad2deg(self, *args, **kwargs): pass
  @abc.abstractmethod
  def ravel(self, *args, **kwargs): pass
  @abc.abstractmethod
  def real(self, *args, **kwargs): pass
  @abc.abstractmethod
  def reciprocal(self, *args, **kwargs): pass
  @abc.abstractmethod
  def remainder(self, *args, **kwargs): pass
  @abc.abstractmethod
  def repeat(self, *args, **kwargs): pass
  @abc.abstractmethod
  def reshape(self, a: NDArrayLike, shape) -> NDArray: pass
  @abc.abstractmethod
  def result_type(self, *args, **kwargs): pass
  @abc.abstractmethod
  def roll(self, *args, **kwargs): pass
  @abc.abstractmethod
  def rot90(self, *args, **kwargs): pass
  @abc.abstractmethod
  def round(self, *args, **kwargs): pass
  @abc.abstractmethod
  def select(self, *args, **kwargs): pass
  @abc.abstractmethod
  def shape(self, a: NDArrayLike): pass
  @abc.abstractmethod
  def sign(self, a: NDArrayLike) -> NDArray: pass
  @abc.abstractmethod
  def signbit(self, *args, **kwargs): pass
  @abc.abstractmethod
  def sin(self, *args, **kwargs): pass
  @abc.abstractmethod
  def sinc(self, *args, **kwargs): pass
  @abc.abstractmethod
  def sinh(self, *args, **kwargs): pass
  @abc.abstractmethod
  def size(self, *args, **kwargs): pass
  @abc.abstractmethod
  def sort(self, *args, **kwargs): pass
  @abc.abstractmethod
  def split(self, *args, **kwargs): pass
  @abc.abstractmethod
  def sqrt(self, *args, **kwargs): pass
  @abc.abstractmethod
  def square(self, *args, **kwargs): pass
  @abc.abstractmethod
  def squeeze(self, *args, **kwargs): pass
  @abc.abstractmethod
  def stack(self, *args, **kwargs): pass
  @abc.abstractmethod
  def std(self, *args, **kwargs): pass
  @abc.abstractmethod
  def subtract(self, a: NDArrayLike, b: NDArrayLike) -> NDArray: pass
  @abc.abstractmethod
  def sum(self, a: NDArrayLike, **kwargs) -> NDArray: pass
  @abc.abstractmethod
  def swapaxes(self, *args, **kwargs): pass
  @abc.abstractmethod
  def take(self, *args, **kwargs): pass
  @abc.abstractmethod
  def take_along_axis(self, *args, **kwargs): pass
  @abc.abstractmethod
  def tan(self, *args, **kwargs): pass
  @abc.abstractmethod
  def tanh(self, *args, **kwargs): pass
  @abc.abstractmethod
  def tensordot(self, a: NDArrayLike, b: NDArrayLike, axes) -> NDArray: pass
  @abc.abstractmethod
  def tile(self, *args, **kwargs): pass
  @abc.abstractmethod
  def trace(self, *args, **kwargs): pass
  @abc.abstractmethod
  def transpose(self, *args, **kwargs): pass
  @abc.abstractmethod
  def tri(self, *args, **kwargs): pass
  @abc.abstractmethod
  def tril(self, *args, **kwargs): pass
  @abc.abstractmethod
  def triu(self, *args, **kwargs): pass
  @abc.abstractmethod
  def true_divide(self, *args, **kwargs): pass
  @abc.abstractmethod
  def uint16(self, *args, **kwargs): pass
  @abc.abstractmethod
  def uint32(self, *args, **kwargs): pass
  @abc.abstractmethod
  def uint64(self, *args, **kwargs): pass
  @abc.abstractmethod
  def uint8(self, *args, **kwargs): pass
  @abc.abstractmethod
  def vander(self, *args, **kwargs): pass
  @abc.abstractmethod
  def var(self, *args, **kwargs): pass
  @abc.abstractmethod
  def vdot(self, *args, **kwargs): pass
  @abc.abstractmethod
  def vsplit(self, *args, **kwargs): pass
  @abc.abstractmethod
  def vstack(self, *args, **kwargs): pass
  @abc.abstractmethod
  def where(self, condition: NDArrayLike, x, y) -> NDArray: pass
  @abc.abstractmethod
  def zeros(self, shape, **kwargs) -> NDArray: pass
  @abc.abstractmethod
  def zeros_like(self, a) -> NDArray: pass

  #
  # Properties.
  #
  @property
  def e(self): raise NotImplementedError()
  @property
  def inf(self): raise NotImplementedError()
  @property
  def newaxis(self): raise NotImplementedError()
  @property
  def pi(self): raise NotImplementedError()
