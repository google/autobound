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

"""The IntervalArithmetic class."""

import functools
from typing import Callable, Sequence, Union

# pylint: disable=g-multiple-import
from autobound.types import (
    Interval, IntervalLike, NDArray, NDArrayLike, NumpyLike)


class IntervalArithmetic:
  """Interval arithmetic on n-dimensional arrays via a numpy-like back end.

  Note: the intervals returned by methods in this class are only correct up to
  floating point roundoff error.
  """

  def __init__(self, np_like: NumpyLike):
    self.np_like = np_like

  def add(self,
          a: Union[NDArrayLike, IntervalLike],
          b: Union[NDArrayLike, IntervalLike]) -> Union[NDArray, Interval]:
    """Returns the sum of two intervals."""
    a_is_interval = isinstance(a, tuple)
    b_is_interval = isinstance(b, tuple)
    if a_is_interval and b_is_interval:
      return (self.np_like.add(a[0], b[0]), self.np_like.add(a[1], b[1]))
    elif a_is_interval:
      return (self.np_like.add(a[0], b), self.np_like.add(a[1], b))
    elif b_is_interval:
      return (self.np_like.add(a, b[0]), self.np_like.add(a, b[1]))
    else:
      return self.np_like.add(a, b)

  def arbitrary_bilinear(
      self,
      a: Union[NDArrayLike, IntervalLike],
      b: Union[NDArrayLike, IntervalLike],
      bilinear: Callable[[NDArrayLike, NDArrayLike], NDArray],
      assume_product: bool = False
  ) -> Union[NDArray, Interval]:
    """Applies a bilinear operation (e.g., matmul, conv2d) to two intervals.

    Args:
      a: an NDArray-like or Interval-like object.
      b: an NDArray-like or Interval-like object.
      bilinear: a callable that takes two NDArray-like arguments, and returns
        the NDArray that results from applying some bilinear operation to them.
      assume_product: if True, we assume that each element of the NDArray
        returned by `bilinear` is a product of some element of `a` and some
        element of `b`, and use a rule that returns a tighter interval under
        this assumption.

    Returns:
      an NDArray or Interval representing the result of applying the bilinear
      operation to `a` and `b`.
    """
    a_is_interval = isinstance(a, tuple)
    b_is_interval = isinstance(b, tuple)
    if not a_is_interval and not b_is_interval:
      return bilinear(a, b)

    if assume_product:
      def yield_endpoint_products():
        a_endpoints = a if a_is_interval else (a,)
        b_endpoints = b if b_is_interval else (b,)
        for a_endpoint in a_endpoints:
          for b_endpoint in b_endpoints:
            yield bilinear(a_endpoint, b_endpoint)  # pytype: disable=wrong-arg-types
      endpoint_products = list(yield_endpoint_products())
      return (
          functools.reduce(self.np_like.minimum, endpoint_products),
          functools.reduce(self.np_like.maximum, endpoint_products)
      )
    else:
      # TODO(mstreeter): there are multiple methods that could be used here,
      # which make different tradeoffs between computation and the tightness of
      # the returned interval.
      # TODO(mstreeter): add reference to proof of correctness for the method
      # used here.
      def positive_and_negative_parts(x):
        return (self.np_like.maximum(0, x), self.np_like.minimum(0, x))

      if not b_is_interval:
        assert a_is_interval
        b_pos, b_neg = positive_and_negative_parts(b)
        min_vals = self.np_like.add(bilinear(a[0], b_pos),
                                    bilinear(a[1], b_neg))
        max_vals = self.np_like.add(bilinear(a[1], b_pos),
                                    bilinear(a[0], b_neg))
        return (min_vals, max_vals)
      elif not a_is_interval:
        a_pos, a_neg = positive_and_negative_parts(a)
        min_vals = self.np_like.add(bilinear(a_pos, b[0]),
                                    bilinear(a_neg, b[1]))
        max_vals = self.np_like.add(bilinear(a_pos, b[1]),
                                    bilinear(a_neg, b[0]))
        return (min_vals, max_vals)
      else:
        assert a_is_interval and b_is_interval
        u, v = a
        w, x = b
        u_pos, u_neg = positive_and_negative_parts(u)
        v_pos, v_neg = positive_and_negative_parts(v)
        w_pos, w_neg = positive_and_negative_parts(w)
        x_pos, x_neg = positive_and_negative_parts(x)
        min_pairs = [(u_pos, w_pos), (v_pos, w_neg),
                     (u_neg, x_pos), (v_neg, x_neg)]
        min_vals = functools.reduce(
            self.np_like.add,
            [bilinear(x, y) for x, y in min_pairs]
        )
        max_pairs = [(v_pos, x_pos), (v_neg, w_pos),
                     (u_pos, x_neg), (u_neg, w_neg)]
        max_vals = functools.reduce(
            self.np_like.add,
            [bilinear(x, y) for x, y in max_pairs]
        )
        return (min_vals, max_vals)

  def as_interval(self, a: IntervalLike) -> Interval:
    return tuple(self.np_like.asarray(c) for c in a)

  def as_interval_or_ndarray(
      self,
      a: Union[NDArrayLike, IntervalLike]) -> Union[NDArray, Interval]:
    if isinstance(a, tuple):
      return self.as_interval(a)
    else:
      return self.np_like.asarray(a)

  def multiply(
      self,
      a: Union[NDArrayLike, IntervalLike],
      b: Union[NDArrayLike, IntervalLike]) -> Union[NDArray, Interval]:
    """Returns the element-wise product of two intervals."""
    return self.arbitrary_bilinear(a, b, self.np_like.multiply, True)

  def ndim(self, a: Union[IntervalLike, NDArrayLike]) -> int:
    return self.np_like.ndim(a[0] if isinstance(a, tuple) else a)

  def negative(self,
               a: Union[NDArrayLike, IntervalLike]) -> Union[NDArray, Interval]:
    """Returns element-wise negative of an interval."""
    if isinstance(a, tuple):
      return (self.np_like.negative(a[1]), self.np_like.negative(a[0]))
    else:
      return self.np_like.negative(a)

  def outer_power(self,
                  a: Union[NDArrayLike, IntervalLike],
                  exponent: int,
                  batch_dims: int = 0) -> Union[NDArray, Interval]:
    """Returns a repeated outer product."""
    if exponent < 0:
      raise ValueError(exponent)
    elif exponent == 0:
      return self.np_like.asarray(1)
    elif self.ndim(a) == 0:
      return self.power(a, exponent)
    else:
      # For the off-diagonal elements of the output, the best we can do is
      # to repeatedly call self.outer_product().  For the diagonal elements,
      # we can use self.power() to get a tighter result.
      a = self.as_interval_or_ndarray(a)
      running_outer_product = a
      for _ in range(exponent - 1):
        running_outer_product = self.outer_product(running_outer_product, a,
                                                   batch_dims)
      if batch_dims != 0:
        # TODO(mstreeter): adapt the code below to handle batch_dims != 0.
        return running_outer_product
      try:
        a_is_interval = isinstance(a, tuple)
        eye = _generalized_diag_ndarray(
            self.np_like.ones_like(a[0] if a_is_interval else a),
            exponent, self.np_like)
      except NotImplementedError:
        return running_outer_product
      diagonal_elements = self.power(a, exponent)
      diag = self._generalized_diag_interval(diagonal_elements, exponent)
      return self.add(
          self.multiply(running_outer_product, 1 - eye),
          diag
      )

  def outer_product(self,
                    a: Union[NDArrayLike, IntervalLike],
                    b: Union[NDArrayLike, IntervalLike],
                    batch_dims: int = 0) -> Union[NDArray, Interval]:
    """Interval variant of _ndarray_outer_product()."""
    if batch_dims > self.ndim(a) or batch_dims > self.ndim(b):
      raise ValueError((self.ndim(a), self.ndim(b), batch_dims))
    product = functools.partial(_ndarray_outer_product,
                                batch_dims=batch_dims, np_like=self.np_like)
    return self.arbitrary_bilinear(a, b, product, True)

  def power(self, a: Union[NDArrayLike, IntervalLike],
            exponent: float) -> Union[NDArray, Interval]:
    """Returns a**exponent (element-wise)."""
    a = self.as_interval_or_ndarray(a)
    a_is_interval = isinstance(a, tuple)
    if a_is_interval:
      if exponent < 0:
        raise NotImplementedError(exponent)
      elif exponent == 0:
        return self.np_like.ones_like(a[0])
      else:
        # For scalars u and v, with u <= v, and even K, the left end point of
        # [u, v]**K is 0 if u <= 0 <= v, and is min{u**K, v**K} otherwise.
        # If K is odd, the left end point is u**K.  The expression for
        # min_vals below handles all cases.
        #
        # The right and point is always max{u**K, v**K}, giving a simpler
        # expression for max_vals.
        contains_zero = self.np_like.logical_and(a[0] < 0, a[1] > 0)
        pow0 = a[0]**exponent
        pow1 = a[1]**exponent
        min_vals = functools.reduce(self.np_like.minimum,
                                    [pow0, pow1, (1-contains_zero)*pow0])
        max_vals = self.np_like.maximum(pow0, pow1)
        return (min_vals, max_vals)
    else:
      return self.np_like.power(a, exponent)

  def shape(self, a: Union[IntervalLike, NDArrayLike]):
    # Note: calling tuple(...) is necessary when self.np_like is
    # tf.experimental.numpy.
    return tuple(self.np_like.shape(a[0] if isinstance(a, tuple) else a))

  def subtract(
      self,
      a: Union[NDArrayLike, IntervalLike],
      b: Union[NDArrayLike, IntervalLike]) -> Union[NDArray, Interval]:
    """Returns the difference between two intervals."""
    a_is_interval = isinstance(a, tuple)
    b_is_interval = isinstance(b, tuple)
    if a_is_interval and b_is_interval:
      return (self.np_like.subtract(a[0], b[1]),
              self.np_like.subtract(a[1], b[0]))
    elif a_is_interval:
      return (self.np_like.subtract(a[0], b), self.np_like.subtract(a[1], b))
    elif b_is_interval:
      return (self.np_like.subtract(a, b[1]), self.np_like.subtract(a, b[0]))
    else:
      return self.np_like.subtract(a, b)

  def tensordot(
      self,
      a: Union[NDArrayLike, IntervalLike],
      b: Union[NDArrayLike, IntervalLike],
      axes) -> Union[NDArray, Interval]:
    """Like np.tensordot(), but for intervals."""
    bilinear = functools.partial(self.np_like.tensordot, axes=axes)
    return self.arbitrary_bilinear(a, b, bilinear, axes == 0)

  def _generalized_diag_interval(
      self,
      a: Union[NDArrayLike, IntervalLike], n: int) -> Union[NDArray, Interval]:
    """Interval variant of _generalized_diag_ndarray."""
    if isinstance(a, tuple):
      if len(a) != 2:
        raise ValueError()
      return (_generalized_diag_ndarray(a[0], n, self.np_like),
              _generalized_diag_ndarray(a[1], n, self.np_like))
    else:
      return _generalized_diag_ndarray(a, n, self.np_like)


def _generalized_diag_ndarray(a: NDArrayLike, n: int,
                              np_like: NumpyLike) -> NDArray:
  """Returns NDArray of shape shape(a)*n, with a on diagonal."""
  if n == 1:
    return np_like.asarray(a)
  elif n == 2:
    a = np_like.asarray(a)
    if a.ndim == 1:
      return np_like.diag(a)
    else:
      raise NotImplementedError(a.ndim)
  else:
    raise NotImplementedError(n)


def _stringify(axes: Sequence[int]) -> str:
  """Helper for creating an einsum() equation string."""
  offset = ord('a')
  return ''.join(chr(i + offset) for i in axes)


def _ndarray_outer_product(a: NDArrayLike,
                           b: NDArrayLike,
                           batch_dims: int,
                           np_like: NumpyLike) -> NDArray:
  """Returns an outer product with batch dimensions.

  Args:
    a: an NDArray-like object.
    b: an NDArray-like object.
    batch_dims: number of batch dimensions.
    np_like: a Numpy-like backend.

  Returns:
    an NDArray c such that, for every tuple I that indexes the first
    batch_dims elements of a (and b), every tuple J that indexes the last
    a.ndim - batch_dims elements of a, and every tuple K that indexes the last
    a.ndim - batch_dims elements of b, we have:

      c[I + J + K] = a[I + J] * b[I + K]
  """
  a = np_like.asarray(a)
  b = np_like.asarray(b)
  if batch_dims == 0:
    return np_like.tensordot(a, b, 0)
  else:
    a_axes = tuple(range(a.ndim))
    b_non_batch_axes = tuple(range(a.ndim, a.ndim + b.ndim - batch_dims))
    b_axes = tuple(range(batch_dims)) + b_non_batch_axes
    output_axes = a_axes + b_non_batch_axes
    eq = (_stringify(a_axes) + ',' + _stringify(b_axes) +
          '->' + _stringify(output_axes))
    return np_like.einsum(eq, a, b)
