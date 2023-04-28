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

"""Library for evaluating different types of polynomials."""

import math
import operator
from typing import Callable, Iterator, Sequence, TypeVar, Union

from autobound import interval_arithmetic
from autobound import types

Foo = TypeVar('Foo')  # some arbitrary type, like NDArray or Interval
FooLike = TypeVar('FooLike', bound=Foo)  # pytype: disable=invalid-typevar


def eval_polynomial(
    coefficients: Sequence[FooLike],
    z: FooLike,
    inner_product: Callable[[FooLike, FooLike], Foo],
    outer_power: Callable[[FooLike, int], Foo],
    add: Callable[[FooLike, FooLike], Foo] = operator.add,
    additive_identity: Foo = 0,
    multiplicative_identity: Foo = 1) -> Foo:
  """Returns the value of a polynomial at a specific point."""
  running_sum = additive_identity
  z_to_the_i = multiplicative_identity
  for i, coefficient in enumerate(coefficients):
    if i > 0:
      z_to_the_i = outer_power(z, i)
    term = inner_product(coefficient, z_to_the_i)
    running_sum = add(running_sum, term)
  return running_sum


def eval_elementwise_taylor_enclosure(
    enclosure: types.ElementwiseTaylorEnclosureLike,
    x_minus_x0: Union[types.NDArrayLike, types.IntervalLike],
    np_like: types.NumpyLike) -> Union[types.Interval, types.NDArray]:
  """Returns value of an ElementwiseTaylorEnclosure at x-x0."""
  set_arithmetic = interval_arithmetic.IntervalArithmetic(np_like)
  return eval_polynomial(enclosure,
                         set_arithmetic.as_interval_or_ndarray(x_minus_x0),
                         set_arithmetic.multiply,
                         set_arithmetic.power,
                         set_arithmetic.add,
                         np_like.array(0),
                         np_like.array(1))


def eval_taylor_enclosure(
    enclosure: types.TaylorEnclosureLike,
    x_minus_x0: Union[types.NDArrayLike, types.IntervalLike],
    np_like: types.NumpyLike) -> Union[types.Interval, types.NDArray]:
  """Returns value of an TaylorEnclosure at x-x0."""
  set_arithmetic = interval_arithmetic.IntervalArithmetic(np_like)
  inner_product = (
      lambda a, b: set_arithmetic.tensordot(a, b, set_arithmetic.ndim(b)))
  return eval_polynomial(enclosure,
                         set_arithmetic.as_interval_or_ndarray(x_minus_x0),
                         inner_product,
                         set_arithmetic.outer_power,
                         set_arithmetic.add,
                         np_like.array(0),
                         np_like.array(1))


def arbitrary_bilinear(
    a: Sequence[FooLike],
    b: Sequence[FooLike],
    add: Callable[[FooLike, FooLike], Foo] = operator.add,
    additive_identity: Foo = 0,
    term_product_coefficient: Callable[[FooLike, FooLike, int, int], Foo]
    = lambda c0, c1, i, j: c0*c1,
) -> tuple[Foo, ...]:
  """Applies an arbitrary bilinear operation to two polynomials.

  The arguments a and b give the coefficients of polynomials, defined in terms
  of some inner product <x, y> and some exponentiation operator:

    P_a(z) = sum_{i=0}^{len(a)-1} <a[i], z**i>.

  Similarly, the sequence b represents a polynomial P_b(z).

  Args:
    a: a polynomial (sequence of coefficients)
    b: a polynomial (sequence of coefficients)
    add: a function that returns the sum of two polynomial coefficients
    additive_identity: a addtive identity object
    term_product_coefficient: a callable that, given arguments c0, c1, i, j,
      returns d such that op(<c0, z**i>, <c1, z**j>) = <d, z**(i+j)>, where op
      is the underlying bilinear operation.

  Returns:
    a polynomial Q (tuple of coefficients), such that for any z,
      op(P_a(z), P_b(z)) == Q(z)
    where op is the underlying bilinear operation.
  """
  # By bilinearity,
  # op(sum_i <a[i], z**i>, sum_j <b[j], z**j>)
  # == sum_{ij} op(<a[i], z**i>, <b[j], z**j>)
  # == sum_{ij} <term_product_coefficient(a[i], b[j], i, j), z**(i+j)>.
  output_degree = len(a) + len(b) - 2
  output = [additive_identity] * (output_degree + 1)
  # If a and b have length n, this takes time O(n^2).  If we ever care about
  # large n, we could consider implementing an O(n log n) algorithm using
  # Fourier transforms.
  for i, c0 in enumerate(a):
    for j, c1 in enumerate(b):
      c = term_product_coefficient(c0, c1, i, j)
      output[i+j] = add(output[i+j], c)
  return tuple(output)


def integer_power(
    a: Sequence[FooLike],
    exponent: int,
    add: Callable[[FooLike, FooLike], Foo] = operator.add,
    additive_identity: Foo = 0,
    multiplicative_identity: Foo = 1,
    term_product_coefficient: Callable[[FooLike, FooLike, int, int], Foo]
    = lambda c0, c1, i, j: c0*c1,
    term_power_coefficient: Callable[[FooLike, int, int], Foo]
    = lambda c, i, j: c**j,
    scalar_product: Callable[[int, FooLike], Foo] = operator.mul
) -> tuple[Foo, ...]:
  """Returns the coefficients of a polynomial raised to a power.

  The arguments a gives the coefficients of a polynomial, defined in terms
  of some inner product <x, y> and some exponentiation operator:

    P_a(z) = sum_{i=0}^{len(a)-1} <a[i], z**i>.

  Let op be some bilinear, associative, and commutative operation.  We define:

    power(a, 0) == multiplicative_identity
    power(a, k) = op(a, power(a, k-1)).

  This code uses the functions provided as arguments to efficiently compute
  the coefficients of the polynomial power(a, exponent).

  When the coefficients of P_a are intervals, this efficient computation
  translates into tighter intervals in the returned coefficients.

  Args:
    a: a polynomial (sequence of coefficients)
    exponent: a non-negative integer exponent
    add: a function that returns the sum of two polynomial coefficients
    additive_identity: a addtive identity object
    multiplicative_identity: a multiplicative identity object
    term_product_coefficient: a callable that, given arguments c0, c1, i, j,
      returns d such that op(<c0, z**i>, <c1, z**j>) = <d, z**(i+j)>, where op
      is the underlying bilinear operation.
    term_power_coefficient: given arguments c, i, and j, returns d such that:
      (c * z**i)**j == d * z**(i*j)
    scalar_product: a callable that, given as arguments a non-negative integer i
      and coefficient c, returns the result of adding c to itself i times.

  Returns:
    the coefficients of the polynomial P_a, raised to the exponent power.
  """
  if exponent < 0:
    raise ValueError(exponent)
  elif exponent == 0:
    return (multiplicative_identity,)
  else:
    # To understand what this code is doing, it is helpful to consider the
    # special case where `a` is a sequence of floats, and all arguments have
    # their default values.  Then, we just need to compute the coefficients
    # of the scalar polynomial:
    #
    #     (a[0] + a[1]*z**1 + ... + a[k-1])**exponent
    #
    # where k = len(a).
    #
    # Using the multinomial theorem, the result is a polynomial whose ith
    # coefficient is:
    #
    #     sum_{p in Partitions(i, exponent, k)}
    #        (exponent choose (p_0, p_1, ..., p_{k-1})) *
    #        Prod_{j=0}^{k-1} a[j]**p_j
    #
    # where Partitions(i, exponent, k) is the set of length-k non-negative
    # integer tuples whose elements sum to `exponent`, and that furthermore
    # satisfy sum_{j=0}^{k-1} j*p_j == i.
    #
    # The code below uses a generalization of this idea that works for an
    # arbitrary commutative and associative bilinear operation (rather than
    # just scalar multiplication).  In the general version, the product
    # series Prod_{j=0}^{k-1} a[j]**p_j is computed via appropriate calls to
    # term_product_coefficient(), term_power_coefficient() and scalar_product().

    def get_coeff(i: int) -> Foo:
      c = additive_identity
      for p in _iter_partitions(i, exponent, len(a)):
        assert sum(p) == exponent
        assert sum(j*p_j for j, p_j in enumerate(p)) == i
        running_product = multiplicative_identity
        running_product_power = 0
        for j, p_j in enumerate(p):
          running_product = term_product_coefficient(
              running_product,
              term_power_coefficient(a[j], j, p_j),
              running_product_power,
              j*p_j
          )
          running_product_power += j*p_j
        assert running_product_power == i
        term = scalar_product(_multinomial_coefficient(p), running_product)
        c = add(c, term)
      return c
    output_degree = (len(a) - 1) * exponent
    return tuple(get_coeff(i) for i in range(1 + output_degree))


def _iter_partitions(
    n: int, m: int, k: int) -> Iterator[tuple[int, ...]]:
  """Yields length-k tuples with sum m and sum_{j=1}^k (j-1)*i_j == n."""
  if n < 0:
    raise ValueError(n)
  if m < 0:
    raise ValueError(m)
  if k <= 0:
    raise ValueError(k)
  if k == 1:
    if n == 0:
      yield (m,)
  else:
    for z in range(min(m+1, n // (k-1) + 1)):
      for p in _iter_partitions(n - (k-1)*z, m - z, k - 1):
        yield p + (z,)


def _multinomial_coefficient(ks: Sequence[int]) -> int:
  """Returns (n choose (ks[0], ks[1], ...)), where n = sum(ks)."""
  if not ks:
    raise ValueError(ks)
  elif len(ks) == 1:
    return 1
  else:
    return math.comb(sum(ks), ks[0]) * _multinomial_coefficient(ks[1:])
