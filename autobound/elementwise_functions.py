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

"""Library that lists properties of various one-dimensional functions."""

import dataclasses
import functools
import math

from typing import Callable, Optional, Sequence
# pylint: disable=g-multiple-import
from autobound.types import (Interval, IntervalLike, NDArray, NDArrayLike,
                             NumpyLike)


@dataclasses.dataclass(eq=True, frozen=True)
class FunctionId:
  """An identifier for an one-dimensional function."""
  # Functions are identified by a unique name and a derivative order.  For
  # example, the third derivative of the sigmoid function has name == 'sigmoid'
  # and derivative_order == 3.
  #
  # In some cases, multiple FunctionIDs refer to the same mathematical function.
  # For example, the sigmoid function can be represented by the FunctionId
  # with name == 'sigmoid' and derivative_order == 0, or by the FunctionId
  # with name == 'softplus' and derivative_order == 1.
  name: str
  derivative_order: int = 0
  # x_min and x_max specify the domain of function.  A value of None represents
  # +/- infinity.
  x_min: Optional[float] = None
  x_max: Optional[float] = None

  def derivative_id(self, order: int):
    """Returns `FunctionID` for order `order` derivative of this function."""
    return FunctionId(self.name, self.derivative_order + order,
                      x_min=self.x_min, x_max=self.x_max)


@dataclasses.dataclass(eq=True, frozen=True)
class FunctionData:
  """An object that lists properties of a one-dimensional function."""
  # The lists of local minima and maxima include all minima/maxima over the
  # domain of the function, in ascending order.
  local_minima: tuple[float, ...]
  local_maxima: tuple[float, ...]
  monotonically_decreasing: bool = False
  monotonically_increasing: bool = False
  even_symmetric: bool = False  # whether f(x) = f(-x) for all x.

  def monotone_over(
      self,
      region: IntervalLike,
      np_like: NumpyLike) -> tuple[NDArray, NDArray]:
    """Returns ndarrays showing whether the function is monotone over `region`.

    Args:
      region: an `Interval`
      np_like: a `NumpyLike` back end.

    Returns:
      a pair of boolean `NDArray`s `(decreasing, increasing)`, where the
      elements of `decreasing` (resp `increasing`) indicate whether the
      function is monotonically decreasing (resp increasing) over the interval
      specified by the corresponding elements of `region`.
    """
    x_min = np_like.asarray(region[0])
    x_max = np_like.asarray(region[1])

    sorted_extrema = sorted(self.local_minima + self.local_maxima)
    decreasing_conditions = []
    increasing_conditions = []
    for i, x in enumerate(sorted_extrema):
      is_minimum = x in self.local_minima
      if i == 0:
        if is_minimum:
          decreasing_conditions.append(x_max <= x)
        else:
          increasing_conditions.append(x_max <= x)
      else:
        prev_x = sorted_extrema[i-1]
        contained = np_like.logical_and(x_min >= prev_x, x_max <= x)
        if is_minimum:
          decreasing_conditions.append(contained)
        else:
          increasing_conditions.append(contained)
      if i == len(sorted_extrema) - 1:
        if is_minimum:
          increasing_conditions.append(x_min >= x)
        else:
          decreasing_conditions.append(x_min >= x)

    decreasing = functools.reduce(
        np_like.logical_or, decreasing_conditions,
        np_like.full(x_min.shape, self.monotonically_decreasing))
    increasing = functools.reduce(
        np_like.logical_or, increasing_conditions,
        np_like.full(x_min.shape, self.monotonically_increasing))
    return decreasing, increasing


# FunctionIds for various elementwise functions.
ABS = FunctionId('abs')
EXP = FunctionId('exp')
LOG = FunctionId('log', x_min=0.)
SIGMOID = FunctionId('sigmoid')
SOFTPLUS = FunctionId('softplus')
SWISH = FunctionId('swish')


def get_function(function_id: FunctionId,
                 np_like: NumpyLike) -> Callable[[NDArray], NDArray]:
  """Returns a callable version of a function with a given `FunctionId`."""
  if function_id.name == ABS.name:
    if function_id.derivative_order == 0:
      return np_like.abs
    elif function_id.derivative_order == 1:
      return np_like.sign
    else:
      raise NotImplementedError(function_id.derivative_order)
  if function_id.name == EXP.name:
    return np_like.exp
  elif function_id.name == LOG.name:
    k = function_id.derivative_order
    if k == 0:
      return np_like.log
    else:
      sign = -1 if k % 2 == 0 else 1
      return lambda x: sign * math.factorial(k-1) * np_like.asarray(x)**-k
  elif function_id.name == SIGMOID.name:
    return functools.partial(_sigmoid_derivative,
                             function_id.derivative_order, np_like=np_like)
  elif function_id.name == SOFTPLUS.name:
    return functools.partial(_softplus_derivative,
                             function_id.derivative_order, np_like=np_like)
  elif function_id.name == SWISH.name:
    return functools.partial(_swish_derivative,
                             function_id.derivative_order, np_like=np_like)
  else:
    raise NotImplementedError(function_id)


def get_function_data(function_id: FunctionId) -> FunctionData:
  """Gets `FunctionData` given `FunctionId`."""
  if function_id.name == EXP.name:
    return FunctionData((), (), monotonically_increasing=True)
  elif function_id.name == LOG.name:
    k = function_id.derivative_order
    return FunctionData(
        # The domain of the log function is (0, infinity).  Over this domain,
        # none of the derivatives have any local extrema.
        (),
        (),
        monotonically_decreasing=(k%2 == 1),
        monotonically_increasing=(k%2 == 0)
    )
  elif function_id.name == SIGMOID.name:
    # Sigmoid is 1st derivative of softplus, so kth derivative of sigmoid is
    # (k+1)st derivative of softplus.
    return get_function_data(
        SOFTPLUS.derivative_id(1 + function_id.derivative_order))
  elif function_id in _FUNCTION_DATA:
    return _FUNCTION_DATA[function_id]
  else:
    raise NotImplementedError(function_id)


def get_taylor_polynomial_coefficients(
    function_id: FunctionId,
    degree: int,
    x0: NDArray,
    np_like: NumpyLike) -> list[NDArray]:
  """Returns the Taylor polynomial coefficients for a given function at `x0`.

  Args:
    function_id: a `FunctionId`
    degree: the degree of the Taylor polynomial whose coefficients we return
    x0: the reference point
    np_like: a `NumpyLike` backend.

  Returns:
    a list of `NDArray`s of Taylor polynomial coefficients, of length
    `degree+1`.
  """
  coefficients = []
  for i in range(degree + 1):
    f_deriv = get_function(function_id.derivative_id(i), np_like)
    coefficients.append(f_deriv(x0) / math.factorial(i))
  return coefficients


def maximum_value(f,
                  x_min: NDArray,
                  x_max: NDArray,
                  local_maxima: Sequence[float],
                  np_like: NumpyLike) -> NDArray:
  """Returns maximum value of `f` over `[x_min, x_max]`."""
  if not local_maxima:
    return np_like.maximum(f(x_min), f(x_max))
  sorted_maxima = list(sorted(local_maxima, key=f, reverse=True))
  x = sorted_maxima[0]
  return np_like.where(
      np_like.logical_and(x_min <= x, x <= x_max),
      f(x),
      maximum_value(f, x_min, x_max, sorted_maxima[1:], np_like))


def minimum_value(f,
                  x_min: NDArray,
                  x_max: NDArray,
                  local_minima: Sequence[float],
                  np_like: NumpyLike) -> NDArray:
  """Returns minimum value of `f` over `[x_min, x_max]`."""
  if not local_minima:
    return np_like.minimum(f(x_min), f(x_max))
  sorted_minima = list(sorted(local_minima, key=f))
  x = sorted_minima[0]
  return np_like.where(
      np_like.logical_and(x_min <= x, x <= x_max),
      f(x),
      minimum_value(f, x_min, x_max, sorted_minima[1:], np_like))


def _get_range(f,
               x_min: NDArray,
               x_max: NDArray,
               local_minima: Sequence[float],
               local_maxima: Sequence[float],
               np_like: NumpyLike) -> tuple[NDArray, NDArray]:
  minval = minimum_value(f, x_min, x_max, local_minima, np_like)
  maxval = maximum_value(f, x_min, x_max, local_maxima, np_like)
  return (minval, maxval)


def get_range(function_id: FunctionId,
              trust_region: Interval,
              np_like: NumpyLike) -> Interval:
  """Returns exact range of specified function over `trust_region`."""
  f = get_function(function_id, np_like)
  function_data = get_function_data(function_id)
  return _get_range(
      f,
      trust_region[0],
      trust_region[1],
      function_data.local_minima,
      function_data.local_maxima,
      np_like
  )


def _sigmoid(x: NDArrayLike, np_like: NumpyLike) -> NDArray:
  return np_like.where(
      x >= 0,
      1 / (1 + np_like.exp(-x)),
      np_like.exp(x) / (1 + np_like.exp(x))
  )


def _sigmoid_derivative(order: int, x: NDArrayLike,
                        np_like: NumpyLike) -> NDArray:
  """Returns the (elementwise) derivative of a specified order."""
  # Note: we could make this work for arbitrary order using autodiff, but we
  # don't because this module is backend-agnostic, and we don't have a way to
  # do autodiff in a backend-agnostic way.
  s = _sigmoid(x, np_like)
  sm = _sigmoid(-x, np_like)
  if order == 0:
    return s
  elif order == 1:
    return s*sm
  elif order == 2:
    return s*sm*(1-2*s)
  elif order == 3:
    return s*sm*((1-2*s)**2 - 2*s*sm)
  elif order == 4:
    return (s*sm*(1-2*s)*((1-2*s)**2 - 2*s*sm) +
            s*sm*(-4*(1-2*s)*s*sm - 2*s*sm*(1-2*s)))
  else:
    raise NotImplementedError(order)


def _softplus(x: NDArrayLike, np_like: NumpyLike) -> NDArray:
  # Avoid overflow for large positive x using:
  # log(1+exp(x)) == log(1+exp(-|x|)) + max(x, 0).
  return np_like.log1p(np_like.exp(-np_like.abs(x))) + np_like.maximum(x, 0)


def _softplus_derivative(order: int, x: NDArrayLike,
                         np_like: NumpyLike) -> NDArray:
  if order == 0:
    return _softplus(x, np_like)
  else:
    return _sigmoid_derivative(order - 1, x, np_like)


def _swish(x: NDArrayLike, np_like: NumpyLike) -> NDArray:
  return x*_sigmoid(x, np_like)


def _swish_derivative(order: int, x: NDArrayLike,
                      np_like: NumpyLike) -> NDArray:
  if order == 0:
    return _swish(x, np_like)
  else:
    # swish(x) = x*sigmoid(x)
    # swish'(x) = sigmoid(x) + x*sigmoid'(x).
    # Inductively,
    # swish^{(k)}(x) = k*sigmoid^({k-1})(x) + x*sigmoid^{(k)}(x).
    return (order*_sigmoid_derivative(order - 1, x, np_like) +
            x*_sigmoid_derivative(order, x, np_like))


# Dict from FunctionId to FunctionData.
_FUNCTION_DATA = {
    # ABS
    ABS: FunctionData((0.,), (), even_symmetric=True),
    ABS.derivative_id(1): FunctionData((), (), monotonically_increasing=True),
    # SOFTPLUS
    SOFTPLUS: FunctionData((), (), monotonically_increasing=True),
    SOFTPLUS.derivative_id(1):
        FunctionData((), (), monotonically_increasing=True),
    SOFTPLUS.derivative_id(2):
        FunctionData((), (0.,), even_symmetric=True),
    SOFTPLUS.derivative_id(3):
        FunctionData((1.3169578969249405,), (-1.3169578969249423,)),
    SOFTPLUS.derivative_id(4):
        FunctionData((0.,), (-2.292431669561122, 2.2924316695611195)),
    SOFTPLUS.derivative_id(5):
        FunctionData((-0.8426329481295408, 3.1443184061547065),
                     (-3.144318406154709, 0.8426329481295388)),
    # SWISH
    SWISH: FunctionData((-1.278464542761141,), ()),
    SWISH.derivative_id(1):
        FunctionData((-2.399357280515326,), (2.399357280515324,)),
    SWISH.derivative_id(2):
        FunctionData((-3.4358409935350243, 3.4358409935350225), (0.,)),
    SWISH.derivative_id(3):
        FunctionData((-4.429235100557346, 1.0319582417807385),
                     (-1.0319582417807402, 4.429235100557342)),
    SWISH.derivative_id(4):
        FunctionData((0.,), (-1.8197756117249821, 1.8197756117249804)),
    SWISH.derivative_id(5):
        FunctionData((-0.7177419231466055, 2.5062894864026024),
                     (-2.506289486402605, 0.7177419231466035)),
}
