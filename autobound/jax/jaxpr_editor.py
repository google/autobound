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

"""Library for editing JAX expressions (Jaxprs)."""

import itertools
import types
from typing import Any, Union

from autobound import graph_editor
import jax


def replace(pattern: jax.core.Jaxpr,
            replacement: jax.core.Jaxpr,
            subject: jax.core.Jaxpr) -> jax.core.Jaxpr:
  """Replace occurrences of a pattern Jaxpr within a subject Jaxpr.

  This method return a `Jaxpr` that is an edited version of `subject`
  in which occurrences of `pattern` have been replaced by `replacement`.
  Occurences of `pattern` must be contiguous sequences of equations in
  `subject`.

  Example usage:

  ```python
  pattern_fun = jnp.exp
  replacement_fun = jnp.log
  subject_fun = lambda x: 2*jnp.exp(x+1)
  to_jaxpr = lambda f: jax.make_jaxpr(f)(0.).jaxpr

  replace(
      to_jaxpr(pattern_fun),
      to_jaxpr(replacement_fun),
      to_jaxpr(subject_fun),
  )  # returns Jaxpr for 2*jnp.log(x+1).
  ```

  Args:
    pattern: a `Jaxpr`
    replacement: a `Jaxpr`
    subject: a `Jaxpr`

  Returns:
    a `Jaxpr` in which occurrences of `pattern` have been replaced by
    `replacement`.
  """
  if len(pattern.invars) != len(replacement.invars):
    raise ValueError()
  if len(pattern.outvars) != len(replacement.outvars):
    raise ValueError()
  if pattern.constvars or replacement.constvars:
    raise NotImplementedError()
  pattern_graph = _jaxpr_to_graph(pattern)
  # When creating the replacement graph, offset the variable counts to keep
  # them unique.
  max_count = max(
      v[1] if v[0] else -1  # pytype: disable=unsupported-operands
      for v in pattern_graph.intermediate_variables()
  )
  replacement_graph = _jaxpr_to_graph(replacement, offset=max_count + 1)
  # In the replacement graph, replace inputs/outputs with those in the pattern
  # graph.
  intermediate_variable_map = {}
  for u, v in itertools.chain(
      zip(replacement_graph.inputs, pattern_graph.inputs),
      zip(replacement_graph.outputs, pattern_graph.outputs),
  ):
    intermediate_variable_map[u] = v
  replacement_operations = [
      e.subs(intermediate_variable_map) for e in replacement_graph.operations
  ]
  hypergraph = graph_editor.replace(
      pattern_graph.operations,
      replacement_operations,
      _jaxpr_to_graph(subject),
      _can_bind,
  )
  return _graph_to_jaxpr(hypergraph)


class _EqnData:

  def __init__(self, primitive, params):
    self.primitive = primitive
    self.params = params

  def __eq__(self, other):
    return (isinstance(other, _EqnData) and
            self.primitive == other.primitive and
            _jaxpr_eqn_params_equiv(self.params, other.params))

  def __hash__(self):
    return hash(self.primitive)


# An intermediate variable is a tuple that either represents a jax.core.Var or
# a jax.core.Literal.
_IntermediateVariable = Union[
    # If the first element of the tuple is True, then the tuple represents
    # a jax.core.Var, and is of the form (True, count, suffix, aval).
    tuple[bool, int, str, jax.core.AbstractValue],
    # If the first element of the tuple is False, then the tuple represents
    # a jax.core.Literal, and is of the form (False, val, aval).
    tuple[bool, Any, jax.core.AbstractValue]
]


def _jaxpr_to_graph(jaxpr: jax.core.Jaxpr,
                    offset: int = 0) -> graph_editor.ComputationGraph:
  """Returns a ComputationGraph that represents a Jaxpr.

  Args:
    jaxpr: a `Jaxpr`
    offset: an offset for the indices that appear in intermediate variables.
      This can be used to ensure uniqueness.

  Returns:
    a ComputationGraph that represents `jaxpr`.
  """

  def get_intermediate_variable(
      var_or_literal: Union[jax.core.Var, jax.core.Literal]
  ) -> _IntermediateVariable:
    if isinstance(var_or_literal, jax.core.Var):
      var = var_or_literal
      return (True, var.count + offset, var.suffix, var.aval)
    elif isinstance(var_or_literal, jax.core.Literal):
      literal = var_or_literal
      return (False, literal.val, literal.aval)
    else:
      raise NotImplementedError()

  operations = []
  for eqn in jaxpr.eqns:
    data = _EqnData(eqn.primitive, eqn.params)
    edge = graph_editor.Operation(
        data,
        [get_intermediate_variable(v) for v in eqn.invars],
        [get_intermediate_variable(v) for v in eqn.outvars]
    )
    operations.append(edge)

  data = [get_intermediate_variable(v) for v in jaxpr.constvars]
  return graph_editor.ComputationGraph(
      [get_intermediate_variable(v) for v in jaxpr.invars],
      [get_intermediate_variable(v) for v in jaxpr.outvars],
      operations,
      data=data
  )


def _graph_to_jaxpr(h: graph_editor.ComputationGraph) -> jax.core.Jaxpr:
  """Returns the Jaxpr represented by a ComputationGraph."""
  count_to_var = {}

  def vertex_to_var_or_literal(vertex):
    if vertex[0]:
      _, count, suffix, aval = vertex
      if count not in count_to_var:
        count_to_var[count] = jax.core.Var(count, suffix, aval)
      return count_to_var[count]
    else:
      _, val, aval = vertex
      return jax.core.Literal(val, aval)

  eqns = []
  for edge in h.operations:
    eqn = jax.core.new_jaxpr_eqn(
        invars=[vertex_to_var_or_literal(u) for u in edge.inputs],
        outvars=[vertex_to_var_or_literal(v) for v in edge.outputs],
        primitive=edge.data.primitive,
        params=edge.data.params,
        effects=set()
    )
    eqns.append(eqn)

  invars = [vertex_to_var_or_literal(v) for v in h.inputs]
  outvars = [vertex_to_var_or_literal(v) for v in h.outputs]
  constvars = [vertex_to_var_or_literal(v) for v in h.data]
  return jax.core.Jaxpr(constvars, invars, outvars, eqns)


def _can_bind(u, v):
  if u[0]:
    return v[0]
  else:
    return (not v[0]) and (u[1] == v[1])


# Set of Jaxpr equation params we ignore for matching purposes.
_JAXPR_EQN_PARAMS_TO_IGNORE = frozenset(['weak_type'])


def _jaxpr_eqn_params_equiv(p0, p1) -> bool:
  """Returns whether two Jaxpr equation params dicts are equivalent."""
  if set(p0.keys()) != set(p1.keys()):
    return False
  for k0, v0 in p0.items():
    if k0 in _JAXPR_EQN_PARAMS_TO_IGNORE:
      continue
    v1 = p1[k0]
    if isinstance(v0, jax.core.ClosedJaxpr):
      # TODO(mstreeter): this could incorrectly return True if v0.consts and
      # v1.consts are different.
      if not _same_jaxpr_up_to_variable_renaming(v0.jaxpr, v1.jaxpr,
                                                 ignore_shape=True):
        return False
    elif isinstance(v0, jax.core.Jaxpr):
      if not _same_jaxpr_up_to_variable_renaming(v0, v1, ignore_shape=True):
        return False
    elif isinstance(v0, types.FunctionType):
      if not isinstance(v1, types.FunctionType):
        return False
    elif v0 != v1:
      return False
  return True


def _match_avals(a0, a1, ignore_shape):
  return a0.dtype == a1.dtype and (ignore_shape or (a0.shape == a1.shape))


def _same_jaxpr_up_to_variable_renaming(j0: jax.core.Jaxpr,
                                        j1: jax.core.Jaxpr,
                                        ignore_shape: bool = False) -> bool:
  """Return whether to Jaxprs are identical up to variable renaming."""
  var_map = {}
  def check(v0, v1):
    if isinstance(v0, jax.core.Literal):
      return (isinstance(v1, jax.core.Literal) and v0.val == v1.val and
              v0.aval == v1.aval)
    elif isinstance(v0, jax.core.Var):
      if v0 not in var_map:
        if _match_avals(v0.aval, v1.aval, ignore_shape):
          var_map[v0] = v1
          return True
        else:
          return False
      else:
        return var_map[v0] == v1
    else:
      raise NotImplementedError(v0)

  if (len(j0.constvars) != len(j1.constvars) or
      len(j0.invars) != len(j1.invars) or
      len(j0.outvars) != len(j1.outvars) or
      len(j0.eqns) != len(j1.eqns)):
    return False

  for v0, v1 in itertools.chain(zip(j0.invars, j1.invars),
                                zip(j0.constvars, j1.constvars),
                                zip(j0.outvars, j1.outvars)):
    if not check(v0, v1):
      return False

  for eq0, eq1 in zip(j0.eqns, j1.eqns):
    if eq0.primitive != eq1.primitive:
      return False
    if not _jaxpr_eqn_params_equiv(eq0.params, eq1.params):
      return False
    if (len(eq0.invars) != len(eq1.invars) or
        len(eq0.outvars) != len(eq1.outvars)):
      return False
    for v0, v1 in zip(eq0.invars, eq1.invars):
      if not check(v0, v1):
        return False
    for v0, v1 in zip(eq0.outvars, eq1.outvars):
      if not check(v0, v1):
        return False

  return True