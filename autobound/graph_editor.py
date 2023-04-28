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

"""A library for editing computation graphs.

This library offers more limited functionality than dedicated pattern-matching
libraries like matchpy (https://github.com/HPAC/matchpy).  However, unlike
matchpy and many other existing libraries, it works directly on computation
graphs rather than on trees.  This makes it easier to handle certain use cases,
like editing Jaxprs.
"""

import dataclasses
import itertools
from typing import Any, Callable, Hashable, Mapping, Optional, Sequence

# An intermediate variable in a computation graph (e.g., representing a tensor).
IntermediateVariable = Hashable


@dataclasses.dataclass
class Operation:
  """An operation (a.k.a. equation) in a computation graph."""
  data: Any  # for example, the operation type (e.g., 'MatMul')
  inputs: Sequence[IntermediateVariable]
  outputs: Sequence[IntermediateVariable]

  def subs(self, mapping: Mapping[IntermediateVariable, IntermediateVariable]):
    return Operation(self.data,
                     [mapping.get(u, u) for u in self.inputs],
                     [mapping.get(v, v) for v in self.outputs])


@dataclasses.dataclass
class ComputationGraph:
  """A computation graph (e.g., representing a Jaxpr or a tf.Graph)."""
  inputs: Sequence[IntermediateVariable]
  outputs: Sequence[IntermediateVariable]
  operations: Sequence[Operation]
  data: Any = None  # Any global data associated with the graph.

  def intermediate_variables(self) -> set[IntermediateVariable]:
    intvars = set(self.inputs)
    intvars.update(self.outputs)
    for op in self.operations:
      intvars.update(op.inputs)
      intvars.update(op.outputs)
    return intvars


def match(
    pattern: Sequence[Operation],
    subject: Sequence[Operation],
    can_bind: Callable[[IntermediateVariable, IntermediateVariable], bool]
) -> Optional[dict[IntermediateVariable, IntermediateVariable]]:
  """Checks whether a pattern matches a subject, and returns mapping if so.

  Args:
    pattern: a sequence of `Operations`
    subject: a sequence of `Operations`
    can_bind: a callable that, given as arguments a pattern
      `IntermediateVariable` `u` and a subject `IntermediateVariable` `v`,
      determines whether `u` can be mapped to `v`.  (This could return `False`,
      for example, if `u `represents a constant and `v` represents a different
      constant.)

  Returns:
    A dict `m` representing a match, or `None` if no match was found.  If the
    return value is not `None`, it maps from pattern `IntermediateVertex` to
    subject `IntermediateVertex`, and satisfies:
    ```python
    [e.subs(m) for e in pattern] == subject
    ```
  """
  if len(pattern) != len(subject):
    return None

  vertex_map = {}  # from pattern vertex to subject vertex
  for p, s in zip(pattern, subject):
    if (len(p.inputs) != len(s.inputs) or len(p.outputs) != len(s.outputs)):
      return None
    if p.data != s.data:
      return None
    for u, v in itertools.chain(zip(p.inputs, s.inputs),
                                zip(p.outputs, s.outputs)):
      if u in vertex_map:
        if vertex_map[u] != v:
          return None
      else:
        if can_bind(u, v):
          vertex_map[u] = v
        else:
          return None
  return vertex_map


def replace(
    pattern: Sequence[Operation],
    replacement: Sequence[Operation],
    subject: ComputationGraph,
    can_bind: Callable[[IntermediateVariable, IntermediateVariable], bool]
) -> ComputationGraph:
  """Perform a search/replace on a ComputationGraph.

  This method greedily replaces occurences of a given operation sequence
  `pattern` with an operation sequence `replacement`.

  Args:
    pattern: a sequence of `Operations`
    replacement: a sequence of `Operations` with which to replace occurrences of
      the pattern.
    subject: a `ComputationGraph`
    can_bind: a `Callable` with the same meaning as the corresponding argument
      to `match()`.

  Returns:
    a `ComputationGraph` with occurrences of `pattern` replaced by
    `replacement`.
  """
  # Note: this could be made more efficient using the Knuth-Morris-Pratt
  # algorithm.
  k = len(pattern)
  output_operations = []
  i = 0
  while i < len(subject.operations):
    subgraph = subject.operations[i:i+k]
    m = match(pattern, subgraph, can_bind)
    if m is not None:
      output_operations.extend([e.subs(m) for e in replacement])
      i += k
    else:
      output_operations.append(subject.operations[i])
      i += 1

  return ComputationGraph(
      subject.inputs,
      subject.outputs,
      output_operations,
      data=subject.data
  )
