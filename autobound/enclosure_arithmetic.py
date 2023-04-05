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

"""Code for performing arithmetic on Taylor enclosures."""

import functools
from typing import Callable, Optional, Union

from autobound import interval_arithmetic
from autobound import polynomials
from autobound import primitive_enclosures
# pylint: disable=g-multiple-import
from autobound.types import (
    ElementwiseTaylorEnclosure, ElementwiseTaylorEnclosureLike, Interval,
    IntervalLike, NDArray, NDArrayLike, NumpyLike, TaylorEnclosure,
    TaylorEnclosureLike)


class TaylorEnclosureArithmetic:
  """Taylor enclosure arithmetic via a numpy-like back end.

  Given a specified trust region, maximum polynomial degree D, and NumpyLike
  back end, this class allows you to perform arithmetic operations on
  TaylorEnclosures, and get back TaylorEnclosures of degree <= D.

  What does it mean to perform arithmetic operations on Taylor enclosures?
  Let op(arg_0(x), arg_1(x)) be some binary operation (such as addition or
  multiplication).  Let enclosure_i be a Taylor enclosure for arg_i:

      arg_i(x) in enclosure_i(x)  for x-x0 in trust_region.

  Then, op(enclosure_0, enclosure_1) is a Taylor enclosure f, such that:

      op(arg_0(x), arg_1(x)) in f(x)  for x-x0 in trust_region.

  Example usage:
    import jax.numpy as jnp
    trust_region = (jnp.zeros((3,)), jnp.array([1, 2, 3]))
    arithmetic = TaylorEnclosureArithmetic(2, trust_region, jnp)
    arithmetic.power(
        TaylorEnclosure((0, 1),
        4
    )  # ==> quadratic TaylorEnclosure of x**4, valid for x in trust_region
  """

  def __init__(self,
               max_degree: int,
               trust_region: IntervalLike,
               np_like: NumpyLike):
    """Initializer.

    Args:
      max_degree: the maximum degree polynomial output by any of the functions
        in this class
      trust_region: an interval that contains x-x0 (not x!).
      np_like: a NumpyLike Module
    """
    if np_like.shape(trust_region[0]) != np_like.shape(trust_region[1]):
      raise ValueError(trust_region)
    if not isinstance(max_degree, int):
      raise ValueError(max_degree)

    self.max_degree = max_degree
    self.set_arithmetic = interval_arithmetic.IntervalArithmetic(np_like)
    self.trust_region = self.set_arithmetic.as_interval(trust_region)
    self.np_like = np_like

  def add(self,
          a: TaylorEnclosureLike,
          b: TaylorEnclosureLike) -> TaylorEnclosure:
    """Returns the (possibly-truncated) sum of two TaylorEnclosures."""
    a = as_taylor_enclosure(a, self.np_like)
    b = as_taylor_enclosure(b, self.np_like)

    def get_coefficient(i):
      if i < len(a) and i < len(b):
        return self.set_arithmetic.add(a[i], b[i])
      elif i < len(a):
        return a[i]
      else:
        return b[i]

    sum_ab = tuple(get_coefficient(i) for i in range(max(len(a), len(b))))
    return self._truncate_if_necessary(sum_ab)

  def arbitrary_bilinear(
      self,
      a: TaylorEnclosureLike,
      b: TaylorEnclosureLike,
      pairwise_batched_bilinear: Callable[[NDArray, NDArray, int, int], NDArray]
  ) -> TaylorEnclosure:
    """Applies an arbitrary bilinear operation to two TaylorEnclosures.

    Args:
      a: a TaylorEnclosure
      b: a TaylorEnclosure
      pairwise_batched_bilinear: a callable that takes parameters (u, v, p, q),
        and returns the result of applying some underlying operation 'bilinear'
        to various pairs of arguments, where the first argument is indexed by
        the last p dimensions of u, and the second argument is indexed by the
        last q dimensions of v.

        In the special case p=q=1,

            pairwise_batched_bilinear(u, v, 1, 1)[..., i, j]

                == bilinear(u[..., i], v[..., j])  for all i, j.

        In the general case, for all tuples I = (i_1, i_2, ..., i_p), and
        J = (j_1, j_2, ..., j_q),

            pairwise_batched_bilinear(u, v, p, q)[(...,) + I + J]

                == bilinear(u[(...,) + I], v[(...,) + J]) .

    Returns:
      a TaylorEnclosure c, such that for x-x0 in self.trust_region:

        arg_0 in a(x) and arg_1 in b(x) ==> bilinear(arg0, arg1) in c(x)

      where 'bilinear' is pairwise_batched_bilinear's underlying operation.
    """
    # Our goal is to compute a Taylor enclosure for bilinear(a, b), where:
    #
    #    a = sum_i <a[i], (x-x0)^i>
    #    b = sum_j <b[j], (x-x0)^j>
    #
    # Here, we use ^ to denote self.set_arithmetic.outer_power, and we use
    # <u, v> to denote self.set_arithmetic.tensordot(u, v, v.ndim).
    #
    # Using bilinearity, we can show:
    #
    #    bilinear(a, b)
    #        == sum_{i, j} bilinear(<a[i], (x-x0)^i>, <b[i], (x-x0)^j>).
    #
    # Thus, it suffices to compute a Taylor enclosure for each term, and then
    # sum those up.
    #
    # How do we compute a Taylor enclosure for each term?  It can be shown that:
    #
    #     bilinear(<a[i], (x-x0)^i>, <b[j], (x-x0)^j>)
    #     == bilinear(pairwise_batched_bilinear(a[i], b[j], i*x_ndim, j*x_ndim),
    #                 (x-x0)^(i+j)) .
    #
    # If i + j <= self.max_degree, we use simply use this formula to return
    # the coefficient for term (i, j).  Otherwise, we make use of the trust
    # region to enclose the left hand side in terms of (x-x0)^self.max_degree.

    x_ndim = self.np_like.ndim(self.trust_region[0])

    def get_term_enclosure_coefficient(
        i: int, j: int) -> Union[NDArray, Interval]:
      r"""Returns coefficient of enclosure for term coming from (a[i], b[j]).

      Args:
        i: an index into a
        j: an index into b

      Returns:
        an NDArray or Interval c, such that:

            bilinear(inner(a[i], z^i), inner(b[j], z^j)) \subseteq inner(c, z^k)

        where k = min(i+j, self.max_degree).
      """
      if i + j <= self.max_degree:
        c0, c1, c0_power, c1_power = (a[i], b[j], i, j)
      else:
        excess_degree = i + j - self.max_degree
        # Distribute the excess degree between the two factors.
        excess_i = min(i, excess_degree)
        excess_j = excess_degree - excess_i
        assert (i - excess_i) + (j - excess_j) == self.max_degree
        c0 = self.set_arithmetic.tensordot(
            a[i],
            self.set_arithmetic.outer_power(self.trust_region, excess_i),
            excess_i * x_ndim
        )
        if excess_j > 0:
          c1 = self.set_arithmetic.tensordot(
              b[j],
              self.set_arithmetic.outer_power(self.trust_region, excess_j),
              excess_j * x_ndim
          )
        else:
          c1 = b[j]
        c0_power = i - excess_i
        c1_power = j - excess_j

      def bilinear(y, z):
        return pairwise_batched_bilinear(y, z, c0_power*x_ndim,
                                         c1_power*x_ndim)

      return self.set_arithmetic.arbitrary_bilinear(c0, c1, bilinear)

    # Sum up the coefficients from each term.
    output_degree = min(self.max_degree, len(a) + len(b) - 2)
    product_coefficients = [self.np_like.asarray(0)] * (output_degree + 1)
    for i in range(len(a)):
      for j in range(len(b)):
        term_coefficient = get_term_enclosure_coefficient(i, j)
        term_degree = min(i+j, self.max_degree)
        product_coefficients[term_degree] = self.set_arithmetic.add(
            product_coefficients[term_degree],
            term_coefficient,
        )

    return TaylorEnclosure(tuple(product_coefficients))

  def compose_enclosures(
      self,
      elementwise_enclosure: ElementwiseTaylorEnclosureLike,
      arg_enclosure: TaylorEnclosureLike) -> TaylorEnclosure:
    """Returns composition of two enclosures."""
    # TODO(mstreeter): we can potentially do something more efficient than
    # computing each term separately and summing the results.
    if not elementwise_enclosure:
      raise ValueError()
    arg_diff_enclosure = (0,) + arg_enclosure[1:]
    output = TaylorEnclosure((self.np_like.array(0),))

    def interval_left_broadcasting_multiply(a, b):
      bilinear = functools.partial(_left_broadcasting_multiply,
                                   np_like=self.np_like)
      return self.set_arithmetic.arbitrary_bilinear(a, b, bilinear,
                                                    assume_product=True)

    for p, coefficient in enumerate(elementwise_enclosure):
      if p == 0:
        term = (coefficient,)
      else:
        poly = self.power(arg_diff_enclosure, p)
        term = tuple(
            # The special-casing when i < p ensures that the TaylorEnclosure
            # returned by this function will not contain trivial intervals
            # the form (x, x).
            #
            # The issue is that the first p elements of 'poly' are guaranteed
            # to be 0, but the expression below can express this as the interval
            # (0, 0).
            0 if i < p else interval_left_broadcasting_multiply(coefficient, t)
            for i, t in enumerate(poly)
        )
      output = self.add(output, term)
    assert output is not None
    return output

  def divide(self,
             a: TaylorEnclosureLike,
             b: TaylorEnclosureLike) -> TaylorEnclosure:
    return self.multiply(a, self.power(b, -1))

  def get_elementwise_fun(
      self,
      get_elementwise_enclosure: Callable[
          [NDArray, Interval, int, NumpyLike],
          ElementwiseTaylorEnclosure
      ]):
    """Returns elementwise function that inputs/output TaylorEnclosures."""
    def fun(
        arg_enclosure: TaylorEnclosureLike,
        arg_trust_region: Optional[Union[NDArrayLike, IntervalLike]] = None
    ) -> TaylorEnclosure:
      if arg_trust_region is None:
        # If arg_trust_region is not provided derive it from arg_enclosure.
        degree_0_enclosure = enclose_enclosure(arg_enclosure, self.trust_region,
                                               0, self.np_like)
        arg_trust_region = degree_0_enclosure[0]
      arg_trust_region = self.set_arithmetic.as_interval_or_ndarray(
          arg_trust_region)
      if not isinstance(arg_trust_region, tuple):
        arg_trust_region = (arg_trust_region, arg_trust_region)
      x0 = arg_enclosure[0]
      if isinstance(x0, tuple):
        assert self.max_degree == 0
        assert len(x0) == 2
        x0 = x0[0]
      elementwise_enclosure = get_elementwise_enclosure(
          x0,
          arg_trust_region,
          self.max_degree,
          self.np_like)
      return self.compose_enclosures(elementwise_enclosure, arg_enclosure)
    return fun

  def multiply(self,
               a: TaylorEnclosureLike,
               b: TaylorEnclosureLike) -> TaylorEnclosure:
    """Returns elementwise product of two TaylorEnclosures."""
    self._validate_taylor_enclosure(a)
    self._validate_taylor_enclosure(b)
    term_product_coefficient = functools.partial(
        _elementwise_term_product_coefficient,
        x_ndim=self.np_like.ndim(self.trust_region[0]),
        np_like=self.np_like)
    product = polynomials.arbitrary_bilinear(
        a,
        b,
        self.set_arithmetic.add,
        self.np_like.asarray(0),
        term_product_coefficient
    )
    return self._truncate_if_necessary(product)

  def negative(self, a: TaylorEnclosureLike) -> TaylorEnclosure:
    return self._truncate_if_necessary(
        TaylorEnclosure(tuple(self.set_arithmetic.negative(c) for c in a)))

  def power(self, a: TaylorEnclosureLike, p: float) -> TaylorEnclosure:
    """Returns a TaylorEnclosure for a**p, of degree <= self.max_degree."""
    self._validate_taylor_enclosure(a)
    if p >= 0 and p == int(p):
      x_ndim = self.np_like.ndim(self.trust_region[0])
      term_product_coefficient = functools.partial(
          _elementwise_term_product_coefficient, x_ndim=x_ndim,
          np_like=self.np_like)
      term_power_coefficient = functools.partial(
          _elementwise_term_power_coefficient, x_ndim=x_ndim,
          np_like=self.np_like)
      multiplicative_identity = self.np_like.ones_like(self.trust_region[0])
      result = polynomials.integer_power(  # pytype: disable=wrong-arg-types
          a,
          p,
          self.set_arithmetic.add,
          self.np_like.asarray(0),
          multiplicative_identity,
          term_product_coefficient,
          term_power_coefficient,
          self.set_arithmetic.multiply
      )
      return self._truncate_if_necessary(result)
    else:
      get_elementwise_enclosure = functools.partial(
          primitive_enclosures.pow_enclosure, p)
      return self.get_elementwise_fun(get_elementwise_enclosure)(a)

  def subtract(self,
               a: TaylorEnclosureLike,
               b: TaylorEnclosureLike) -> TaylorEnclosure:
    return self.add(a, self.negative(b))

  def _truncate_if_necessary(self, a: TaylorEnclosureLike) -> TaylorEnclosure:
    return enclose_enclosure(a, self.trust_region, self.max_degree,
                             self.np_like)

  def _validate_taylor_enclosure(self, a: TaylorEnclosureLike):
    x_shape = self.set_arithmetic.shape(self.trust_region)
    for i, coeff in enumerate(a):
      s = self.set_arithmetic.shape(coeff)
      if s[len(s)-i*len(x_shape):] != i*x_shape:
        raise ValueError(x_shape, i, s, coeff, a)


def as_taylor_enclosure(a: TaylorEnclosureLike,
                        np_like: NumpyLike) -> TaylorEnclosure:
  set_arithmetic = interval_arithmetic.IntervalArithmetic(np_like)
  return TaylorEnclosure(
      tuple(set_arithmetic.as_interval_or_ndarray(c) for c in a))


def enclose_enclosure(
    enclosure: TaylorEnclosureLike,
    trust_region: IntervalLike,
    max_degree: int,
    np_like: NumpyLike,
) -> TaylorEnclosure:
  """Returns a (possibly) lower-degree enclosure of a given TaylorEnclosure."""
  set_arithmetic = interval_arithmetic.IntervalArithmetic(np_like)
  trust_region = set_arithmetic.as_interval(trust_region)
  enclosure = as_taylor_enclosure(enclosure, np_like)
  orig_degree = len(enclosure) - 1
  if orig_degree <= max_degree:
    return enclosure
  else:
    new_final_coefficient = polynomials.eval_taylor_enclosure(
        enclosure[max_degree:], trust_region, set_arithmetic.np_like)
    return TaylorEnclosure(enclosure[:max_degree] + (new_final_coefficient,))


def expand_multiple_dims(a: NDArray, n: int, axis=None) -> NDArray:
  """Like expand_dims, but adds n dims rather than just 1."""
  if axis is None:
    axis = a.ndim
  colon = slice(None, None, None)
  return a[(colon,) * axis + (None,) * n + (...,)]


def map_over_enclosure(
    a: TaylorEnclosure,
    fun: Callable[[NDArray], NDArray]) -> TaylorEnclosure:
  """Apply a function to each NDArray in a TaylorEnclosure."""
  return TaylorEnclosure(
      tuple(tuple(map(fun, c)) if isinstance(c, tuple) else fun(c) for c in a))


def _elementwise_term_power_coefficient(
    c: Union[NDArrayLike, IntervalLike],
    i: int,
    exponent: int,
    x_ndim: int,
    np_like: NumpyLike) -> Union[NDArray, Interval]:
  """Returns d such that <c, z**i>^exponent == <d, z**(i*exponent)>.

  Args:
    c: a coefficient
    i: a non-negative integer
    exponent: a non-negative integer
    x_ndim: the number of dimensions in the independent variable
    np_like: a Numpy-like backend

  Returns:
    an NDArray or Interval d, such that
        <c, z**i>^exponent == <d, z**(i*exponent)>.
    where ** denotes outer product, and ^ denotes elementwise exponentiation.
  """
  set_arithmetic = interval_arithmetic.IntervalArithmetic(np_like)
  batch_dims = set_arithmetic.ndim(c) - i*x_ndim
  if batch_dims < 0:
    raise ValueError((set_arithmetic.ndim(c), i, x_ndim))
  return set_arithmetic.outer_power(c, exponent, batch_dims)


def _elementwise_term_product_coefficient(
    c0: Union[NDArrayLike, IntervalLike],
    c1: Union[NDArrayLike, IntervalLike],
    i: int,
    j: int,
    x_ndim: int,
    np_like: NumpyLike) -> Union[NDArray, Interval]:
  """Returns d such that <c0, z**i> * <c1, z**j> == <d, z**(i+j)>."""
  def product(u, v):
    return _pairwise_batched_multiply(u, v, i*x_ndim, j*x_ndim, np_like)
  set_arithmetic = interval_arithmetic.IntervalArithmetic(np_like)
  return set_arithmetic.arbitrary_bilinear(c0, c1, product, assume_product=True)


def _pairwise_batched_multiply(
    u: NDArrayLike,
    v: NDArrayLike,
    p: int,
    q: int,
    np_like: NumpyLike) -> NDArray:
  """Batched version of multiply, for use as input to arbitrary_bilinear().

  See the docstring for TaylorEnclosureArithmetic.arbitrary_bilinear for
  context.

  Args:
    u: an NDArray of dimension at least p
    v: an NDArray of dimension at least q
    p: a non-negative integer
    q: a non-negative integer
    np_like: a NumpyLike back end

  Returns:
    an NDArray 'output', such that for every pair of tuples
    I = (i_1, i_2, ..., i_p) and J = (j_1, j_2, ..., j_q),

      output[(...,) + I + J] = u[(...,) + I] * v[(...,) + J] .
  """
  u = np_like.asarray(u)
  v = np_like.asarray(v)
  return expand_multiple_dims(u, q) * expand_multiple_dims(v, p, v.ndim-q)


def _left_broadcasting_multiply(a: NDArrayLike, b: NDArrayLike,
                                np_like: NumpyLike) -> NDArray:
  """Multiplies a and b, broadcasting over leftmost dimensions."""
  a = np_like.asarray(a)
  b = np_like.asarray(b)
  if a.ndim > b.ndim:
    raise NotImplementedError()
  return expand_multiple_dims(a, b.ndim - a.ndim) * b
