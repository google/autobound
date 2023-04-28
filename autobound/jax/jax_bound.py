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

"""Code for computing Taylor enclosures in JAX."""

import dataclasses
import functools
from typing import Callable, Optional, Union

from autobound import enclosure_arithmetic
from autobound import interval_arithmetic
from autobound import polynomials
from autobound import primitive_enclosures
from autobound import types
from autobound.jax import jaxpr_editor
import jax
import jax.numpy as jnp


@dataclasses.dataclass
class TaylorBounds:
  """Upper and lower bounds on a function, valid over a trust region."""
  f: Callable[[jnp.array], jnp.array]
  x0: jnp.ndarray
  # Interval containing values of x (not x-x0) for which the bound on f(x)
  # holds.
  x_trust_region: tuple[jnp.array, jnp.array]
  coefficients: types.TaylorEnclosure

  def __call__(self,
               x: types.NDArrayLike) -> Union[types.NDArray, types.Interval]:
    x = jnp.asarray(x)
    return polynomials.eval_taylor_enclosure(self.coefficients, x-self.x0, jnp)

  def final_interval(self) -> tuple[jnp.array, jnp.array]:
    """Returns final coefficient (as a trivial interval, if it is scalar)."""
    c = self.coefficients[-1]
    return c if isinstance(c, tuple) else (c, c)

  def lower(self, x):
    bound = self(x)
    return bound[0] if isinstance(bound, tuple) else bound

  def upper(self, x):
    bound = self(x)
    return bound[1] if isinstance(bound, tuple) else bound


def taylor_bounds(
    f: Callable[[jnp.array], jnp.array],
    max_degree: int,
    propagate_trust_regions: bool = False,
) -> Callable[[jnp.array, tuple[jnp.array, jnp.array]], TaylorBounds]:
  """Returns version of f that returns a TaylorBounds object.

  Args:
    f: a function that takes a jnp.array as input, and returns a jnp.array
    max_degree: the maximum degree TaylorEnclosure for the returned function
      to return
    propagate_trust_regions: if True, trust regions are propagated
      through the Jaxpr, rather than being computed from higher-degree
      enclosures.  This results in tighter bounds at the cost of additional
      memory.

  Returns:
    a function that takes as input a jnp.array x0, and a trust region
    (min_vals, max_vals), and return a TaylorBounds object `bound` such that
    `bound.coefficients` is a TaylorEnclosure g of degree at most max_degree,
    such that:

        f(x) in g(x-x0) for all x with min_vals <= x <= max_vals
  """
  if max_degree < 0:
    raise ValueError(max_degree)
  if max_degree == 0:
    propagate_trust_regions = False  # avoid redundant computation

  jaxpr_factory = jax.make_jaxpr(f)
  def bound_fun(x0: jnp.array,
                x_trust_region: types.Interval) -> TaylorBounds:
    trust_region = interval_arithmetic.IntervalArithmetic(jnp).subtract(
        x_trust_region, x0)

    arithmetic = enclosure_arithmetic.TaylorEnclosureArithmetic(
        max_degree, trust_region, jnp)
    primitive_to_enclosure_fun = _pushforward_funs(arithmetic)

    degree_0_arithmetic = (
        enclosure_arithmetic.TaylorEnclosureArithmetic(0, trust_region, jnp))
    primitive_to_enclosure_fun0 = _pushforward_funs(degree_0_arithmetic)

    closed_jaxpr = jaxpr_factory(x0)
    jaxpr = _rewrite_jaxpr(closed_jaxpr.jaxpr)

    x0 = jnp.asarray(x0)
    if x0.ndim == 0:
      identity = jnp.asarray(1.)
    elif x0.ndim == 1:
      identity = jnp.eye(x0.shape[0])
    else:
      raise NotImplementedError(x0.ndim)
    x0_enclosure = types.TaylorEnclosure(
        (x0, identity) if max_degree > 0 else (x_trust_region,))
    assert len(closed_jaxpr.consts) == len(jaxpr.constvars)
    var_to_intermediate = {
        var: _constant_intermediate_enclosure(val)
        for var, val in zip(jaxpr.constvars, closed_jaxpr.consts)
    }
    assert len(jaxpr.invars) == 1
    var_to_intermediate[jaxpr.invars[0]] = _IntermediateEnclosure(
        enclosure=x0_enclosure,
        trust_region=x_trust_region if propagate_trust_regions else None
    )

    def get_intermediate(
        invar: Union[jax.core.Var, jax.core.Literal]) -> _IntermediateEnclosure:
      if isinstance(invar, jax.core.Var):
        return var_to_intermediate[invar]
      else:
        assert isinstance(invar, jax.core.Literal)
        return _constant_intermediate_enclosure(invar.val)

    for eqn in jaxpr.eqns:
      invar_intermediates = [get_intermediate(invar) for invar in eqn.invars]
      has_non_constant_invars = any(not intermediate.is_constant()
                                    for intermediate in invar_intermediates)
      if has_non_constant_invars:
        fun = primitive_to_enclosure_fun.get(eqn.primitive)
        if fun is None:
          raise NotImplementedError(eqn.primitive)
        outvar_enclosures = fun(*invar_intermediates, **eqn.params)
        if len(eqn.outvars) == 1:
          outvar_enclosures = (outvar_enclosures,)
        if propagate_trust_regions:
          fun0 = primitive_to_enclosure_fun0.get(eqn.primitive)
          assert fun0 is not None
          assert all(i.trust_region is not None for i in invar_intermediates)
          invar_degree_0_intermediates = [
              _IntermediateEnclosure(
                  enclosure=types.TaylorEnclosure((intermediate.trust_region,)))
              for intermediate in invar_intermediates
          ]
          outvar_degree_0_enclosures_a = fun0(*invar_degree_0_intermediates,
                                              **eqn.params)
          if len(eqn.outvars) == 1:
            outvar_degree_0_enclosures_a = [outvar_degree_0_enclosures_a]
          assert len(outvar_degree_0_enclosures_a) == len(outvar_enclosures)
          outvar_degree_0_enclosures_b = [
              enclosure_arithmetic.enclose_enclosure(enclosure, trust_region,
                                                     0, jnp)
              for enclosure in outvar_enclosures
          ]
          outvar_trust_regions = [
              _intersect_intervals(a[0], b[0])
              for a, b in zip(outvar_degree_0_enclosures_a,
                              outvar_degree_0_enclosures_b)
          ]
          for i, (a, b) in enumerate(outvar_trust_regions):
            # It should always be the case that the actual value of the ith
            # output of a function (y0 below) is inside the associated trust
            # region.  But this invariant may not hold due to floating
            # point roundoff error, so we enforce it here.
            #
            # TODO(mstreeter): add a test case that fails if we remove this.
            y0 = outvar_enclosures[i][0]
            outvar_trust_regions[i] = (jnp.minimum(y0, a), jnp.maximum(y0, b))  # pytype: disable=wrong-arg-types
        else:
          outvar_trust_regions = (None,) * len(outvar_enclosures)
        assert all(isinstance(v, tuple) for v in outvar_enclosures), (
            eqn.primitive, fun, outvar_enclosures)
        outvar_intermediates = tuple(
            _IntermediateEnclosure(enclosure=e, trust_region=r)
            for r, e in zip(outvar_trust_regions, outvar_enclosures)
        )
      else:
        invar_values = tuple(intermediate.constant_value()
                             for intermediate in invar_intermediates)
        vals = eqn.primitive.bind(*invar_values, **eqn.params)
        if len(eqn.outvars) == 1:
          vals = (vals,)
        outvar_intermediates = [_constant_intermediate_enclosure(v)
                                for v in vals]

      assert len(outvar_intermediates) == len(eqn.outvars), (
          eqn.primitive, len(outvar_intermediates), len(eqn.outvars))
      for var, intermediate in zip(eqn.outvars, outvar_intermediates):
        if var.count == -1:
          continue  # skip unused output variables
        assert var not in var_to_intermediate
        assert isinstance(intermediate.enclosure, tuple), (
            eqn.primitive, intermediate)
        _validate_taylor_enclosure(intermediate.enclosure, x0.shape)
        var_to_intermediate[var] = intermediate

    assert len(jaxpr.outvars) == 1
    output_intermediate = get_intermediate(jaxpr.outvars[0])
    return TaylorBounds(f=f, x0=x0, x_trust_region=x_trust_region,
                        coefficients=output_intermediate.enclosure)

  return bound_fun


# Type for functions that generate enclosures for primitive elementwise
# functions.  The callable takes arguments x0, trust_region, degree, and
# np_like, and returns an ElementwiseTaylorEnclosure (see examples in
# primitive_enclosures.py).
ElementwiseEnclosureGeneratingFunction = Callable[
    [types.NDArray, types.Interval, int, types.NumpyLike],
    types.ElementwiseTaylorEnclosure
]


# TODO(mstreeter): add a mechanism for supporting a new elementwise function
# given its FunctionData.
def register_elementwise_primitive(
    p: jax.core.Primitive,
    get_enclosure: ElementwiseEnclosureGeneratingFunction):
  """Register an enclosure-generating function for a user-defined primitive.

  Args:
    p: a jax.core.Primitive
    get_enclosure: an ElementwiseEnclosureGeneratingFunction for p.
  """
  _ELEMENTWISE_PRIMITIVE_ENCLOSURES[p] = get_enclosure


#
# Private variables/functions.
#


_PRIMITIVE_NAMES = set()  # type: set[str]
# Rewrite rules are callables that return (pattern Jaxpr, replacement Jaxpr)
# pairs.  We make them callables because the Jaxpr returned by jax.make_jaxpr
# depends on how Jax is configured (in particular whether float64 is enabled),
# and we need to use the Jaxprs that match whatever configuration is being
# used when the rule is applied.
_JAXPR_REWRITE_RULES = [
]  # type: list[Callable[[], tuple[jax.core.Jaxpr, jax.core.Jaxpr]]]


def _register_elementwise_function(
    f: Callable[[jnp.array], jnp.array],
    get_enclosure: ElementwiseEnclosureGeneratingFunction
):
  """Register enclosure-generating function for elementwise Jax function."""
  name = f'__autobound_{f.__name__}__'
  if name in _PRIMITIVE_NAMES:
    raise ValueError(f)
  _PRIMITIVE_NAMES.add(name)
  p = jax.core.Primitive(name)
  p.def_abstract_eval(
      lambda x: jax.abstract_arrays.ShapedArray(x.shape, x.dtype))
  rule = lambda: (jax.make_jaxpr(f)(0.).jaxpr, jax.make_jaxpr(p.bind)(0.).jaxpr)
  _JAXPR_REWRITE_RULES.append(rule)
  register_elementwise_primitive(p, get_enclosure)


# Dict from Jax Primitive to ElementwiseEnclosureGeneratingFunction.
# TODO(mstreeter): support more elementwise functions.
_ELEMENTWISE_PRIMITIVE_ENCLOSURES = {
    jax.lax.abs_p: primitive_enclosures.abs_enclosure,
    jax.lax.exp_p: primitive_enclosures.exp_enclosure,
    jax.lax.log_p: primitive_enclosures.log_enclosure,
}
_register_elementwise_function(jax.nn.sigmoid,
                               primitive_enclosures.sigmoid_enclosure)
_register_elementwise_function(jax.nn.softplus,
                               primitive_enclosures.softplus_enclosure)
_register_elementwise_function(jax.nn.swish,
                               primitive_enclosures.swish_enclosure)


# Set of primitives that can be applied separately to each coefficient of a
# TaylorEnclosure.
_PASS_THRU_PRIMITIVES = frozenset([
    jax.lax.convert_element_type_p,
    jax.lax.reshape_p,
    jax.lax.reduce_sum_p,
    jax.lax.reduce_window_sum_p,
    jax.lax.squeeze_p,
    jax.lax.transpose_p,
    # TODO(mstreeter): add more of these
])


def _rewrite_jaxpr(jaxpr: jax.core.Jaxpr) -> jax.core.Jaxpr:
  """Rewrite a Jaxpr to make is suitable for use by taylor_bounds()."""
  for rule_generator in _JAXPR_REWRITE_RULES:
    pattern, replacement = rule_generator()
    jaxpr = jaxpr_editor.replace(pattern, replacement, jaxpr)
  return jaxpr


@dataclasses.dataclass
class _IntermediateEnclosure:
  """An enclosure for some intermediate variable in a Jaxpr."""
  enclosure: types.TaylorEnclosure
  trust_region: Optional[types.Interval] = None

  def is_constant(self) -> bool:
    """Returns whether self.enclosure represents a constant value."""
    return len(self.enclosure) == 1 and not isinstance(self.enclosure[0], tuple)

  def constant_value(self) -> types.NDArray:
    if not self.is_constant():
      raise ValueError()
    else:
      return self.enclosure[0]  # pytype: disable=bad-return-type


def _broadcast_in_dim_pushforward_fun(intermediate, shape,
                                      broadcast_dimensions):
  """Enclosure-generating function for jax.lax.broadcast_in_dim."""
  enclosure = intermediate.enclosure
  x0 = enclosure[0]
  if isinstance(x0, tuple):
    x0 = enclosure[0][0]
  x_shape = (() if len(enclosure) == 1 else enclosure[1].shape[x0.ndim:])
  def broadcast_ndarray(a, i):
    return jax.lax.broadcast_in_dim(a, shape + i*x_shape, broadcast_dimensions)
  def broadcast_ndarray_or_interval(a, i):
    if isinstance(a, tuple):
      return tuple(broadcast_ndarray(x, i) for x in a)
    else:
      return broadcast_ndarray(a, i)
  return tuple(
      broadcast_ndarray_or_interval(coeff, i)
      for i, coeff in enumerate(enclosure)
  )


def _constant_intermediate_enclosure(val: types.NDArray):
  return _IntermediateEnclosure(enclosure=types.TaylorEnclosure((val,)),
                                trust_region=(val, val))


def _conv_general_dilated_pushforward_fun(arithmetic):
  """Returns function that implements conv_general_dilated on enclosures."""
  def fun(lhs_intermediate: _IntermediateEnclosure,
          rhs_intermediate: _IntermediateEnclosure,
          **params):
    def pairwise_batched_bilinear(a: jnp.array, b: jnp.array,
                                  p: int, q: int) -> jnp.array:
      def move_last_n_dims_to_front(x: jnp.array, n: int):
        if n == 0:
          return x
        perm = tuple(range(x.ndim - n, x.ndim)) + tuple(range(x.ndim - n))
        transposed = jnp.transpose(x, axes=perm)
        return jnp.reshape(transposed, (-1,) + x.shape[:x.ndim-n])

      a_reshaped = move_last_n_dims_to_front(a, p)
      b_reshaped = move_last_n_dims_to_front(b, q)

      c = jax.lax.conv_general_dilated_p.bind(a_reshaped, b_reshaped, **params)
      if p == 0 and q == 0:
        return c
      elif p == 0 or q == 0:
        raise NotImplementedError((p, q))
      c_perm = tuple(range(2, c.ndim)) + (0, 1)
      c_transposed = jnp.transpose(c, axes=c_perm)
      return jnp.reshape(c_transposed,
                         c.shape[2:] + a.shape[a.ndim-p:] + b.shape[b.ndim-q:])

    return arithmetic.arbitrary_bilinear(
        lhs_intermediate.enclosure,
        rhs_intermediate.enclosure,
        pairwise_batched_bilinear)
  return fun


def _dot_general_pushforward_fun(arithmetic):
  """Returns function that implements dot_general on enclosures."""
  def fun(lhs_intermediate: _IntermediateEnclosure,
          rhs_intermediate: _IntermediateEnclosure,
          **params):
    a_contracting_dims = set(a  # pylint: disable=g-complex-comprehension
                             for t in params['dimension_numbers']
                             for a in t[0])
    def pairwise_batched_bilinear(a: jnp.array, b: jnp.array,
                                  p: int, q: int) -> jnp.array:
      transposed_output = jax.lax.dot_general_p.bind(a, b, **params)
      p_start = a.ndim - p - len(a_contracting_dims)
      assert p_start >= 0
      n = transposed_output.ndim
      # Shift axes p_start through p_start+p to the right, so that they start
      # at position n-q.
      assert p_start + p <= n-q, (p_start, p, q, n)
      perm = (
          tuple(range(p_start)) +
          tuple(range(p_start+p, n-q)) +
          tuple(range(p_start, p_start + p)) +
          tuple(range(n-q, n))
      )
      assert len(set(perm)) == n, (p_start, p, q, n, perm)
      return jnp.transpose(transposed_output, axes=perm)
    return arithmetic.arbitrary_bilinear(
        lhs_intermediate.enclosure,
        rhs_intermediate.enclosure,
        pairwise_batched_bilinear
    )
  return fun


def _elementwise_pushforward_fun(arithmetic, get_enclosure):
  f = arithmetic.get_elementwise_fun(get_enclosure)
  def g(intermediate):
    return f(intermediate.enclosure, intermediate.trust_region)
  return g


def _intersect_intervals(
    a: types.Interval, b: types.Interval) -> types.Interval:
  if not len(a) == len(b) == 2:
    raise ValueError()
  return (jnp.maximum(a[0], b[0]), jnp.minimum(a[1], b[1]))  # pytype: disable=wrong-arg-types


def _pass_thru_pushforward_fun(primitive):
  def fun(intermediate, **params):
    return enclosure_arithmetic.map_over_enclosure(
        intermediate.enclosure,
        functools.partial(primitive.bind, **params)
    )
  return fun


# A pushforward function for an underlying primitive with K inputs and N
# outputs takes K _IntermediateEnclosure as arguments, plus kwargs for any
# parameters the primitive has, and returns a tuple of N TaylorEnclosures (or
# in the special case N=1, a single TaylorEnclosure rather than a tuple).
PushforwardFunction = Callable[
    ...,
    Union[types.TaylorEnclosure, tuple[types.TaylorEnclosure, ...]]
]


def _pushforward_funs(
    arithmetic: enclosure_arithmetic.TaylorEnclosureArithmetic
) -> dict[jax.core.Primitive, PushforwardFunction]:
  """Returns dict from primitive to function that inputs/outputs enclosures."""
  def pushforward_integer_pow(intermediate, y: int):
    return arithmetic.power(intermediate.enclosure, y)

  def pushforward_pow(intermediate_0, intermediate_1):
    if not intermediate_1.is_constant():
      raise NotImplementedError()
    exponent = float(intermediate_1.constant_value())
    return arithmetic.power(intermediate_0.enclosure, exponent)

  def wrap(f):
    def g(*args):
      return f(*[intermediate.enclosure for intermediate in args])
    return g

  primitive_to_enclosure_fun = {
      jax.lax.add_p: wrap(arithmetic.add),
      jax.lax.div_p: wrap(arithmetic.divide),
      jax.lax.integer_pow_p: pushforward_integer_pow,
      jax.lax.mul_p: wrap(arithmetic.multiply),
      jax.lax.neg_p: wrap(arithmetic.negative),
      jax.lax.pow_p: pushforward_pow,
      jax.lax.sub_p: wrap(arithmetic.subtract),
      # TODO(mstreeter): handle all bilinear primitives in a uniform way.
      jax.lax.dot_general_p: _dot_general_pushforward_fun(arithmetic),
      jax.lax.conv_general_dilated_p: _conv_general_dilated_pushforward_fun(
          arithmetic),
      jax.lax.broadcast_in_dim_p: _broadcast_in_dim_pushforward_fun,
  }
  primitive_to_enclosure_fun.update({
      primitive: _elementwise_pushforward_fun(arithmetic, get_enclosure)
      for primitive, get_enclosure in _ELEMENTWISE_PRIMITIVE_ENCLOSURES.items()
  })
  primitive_to_enclosure_fun.update({
      primitive: _pass_thru_pushforward_fun(primitive)
      for primitive in _PASS_THRU_PRIMITIVES
  })
  return primitive_to_enclosure_fun


def _validate_taylor_enclosure(a: types.TaylorEnclosureLike, x_shape):
  set_arithmetic = interval_arithmetic.IntervalArithmetic(jnp)
  for i, coeff in enumerate(a):
    s = set_arithmetic.shape(coeff)
    if s[len(s)-i*len(x_shape):] != i*x_shape:
      raise ValueError(x_shape, i, s, coeff, a)
