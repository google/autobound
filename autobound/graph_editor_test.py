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

from absl.testing import absltest
from absl.testing import parameterized
from autobound import graph_editor
from autobound.graph_editor import ComputationGraph, Operation


class TestCase(parameterized.TestCase):

  @parameterized.parameters(
      (
          Operation('foo', ['a', 'b'], ['x', 'y']),
          {'a': 'A', 'y': 'Y'},
          Operation('foo', ['A', 'b'], ['x', 'Y']),
      )
  )
  def test_edge_subs(self, edge, mapping, expected):
    actual = edge.subs(mapping)
    self.assertEqual(expected, actual)

  @parameterized.parameters(
      (
          ComputationGraph(['a', 'b'], ['z'],
                           [Operation('foo', ['a'], ['y']),
                            Operation('bar', ['y'], ['z'])]),
          {'a', 'b', 'y', 'z'}
      ),
  )
  def test_intermediate_variables(self, graph, expected):
    actual = graph.intermediate_variables()
    self.assertSetEqual(expected, actual)

  @parameterized.parameters(
      (
          [Operation('foo', ['x'], ['y'])],
          [Operation('foo', ['x'], ['y'])],
          lambda u, v: True,
          {'x': 'x', 'y': 'y'}
      ),
      (
          [Operation('foo', ['x'], ['y'])],
          [Operation('foo', ['x'], ['y'])],
          lambda u, v: False,
          None
      ),
      (
          [Operation('foo', ['x'], ['y'])],
          [Operation('foo', ['a'], ['b'])],
          lambda u, v: True,
          {'x': 'a', 'y': 'b'}
      ),
      (
          [Operation('foo', ['x'], ['y'])],
          [Operation('bar', ['x'], ['y'])],
          lambda u, v: True,
          None
      ),
  )
  def test_match(self, pattern, subject, can_bind, expected):
    actual = graph_editor.match(pattern, subject, can_bind)
    self.assertEqual(expected, actual)
    if actual is not None:
      self.assertTrue(all(can_bind(u, v) for u, v in actual.items()))
      self.assertEqual([e.subs(actual) for e in pattern], subject)

  @parameterized.parameters(
      (
          [Operation('foo', ['x'], ['y'])],
          [Operation('bar', ['x'], ['y'])],
          ComputationGraph(['a'], ['b'], [Operation('foo', ['a'], ['b'])]),
          lambda u, v: True,
          ComputationGraph(['a'], ['b'], [Operation('bar', ['a'], ['b'])]),
      ),
  )
  def test_replace(self, pattern, replacement, subject, can_bind, expected):
    actual = graph_editor.replace(pattern, replacement, subject, can_bind)
    self.assertEqual(expected, actual)


if __name__ == '__main__':
  absltest.main()
