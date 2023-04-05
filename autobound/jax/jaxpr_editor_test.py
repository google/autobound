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
from autobound.jax import jaxpr_editor
import jax
import jax.numpy as jnp
import numpy as np


softplus_p = jax.core.Primitive('__autobound_softplus__')
softplus_p.def_abstract_eval(
    lambda x: jax.abstract_arrays.ShapedArray(x.shape, x.dtype, weak_type=True))


class TestCase(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'exp',
          jax.make_jaxpr(jnp.exp)(0.).jaxpr,
          jax.make_jaxpr(jnp.exp)(0.).jaxpr,
          False,
          True
      ),
      (
          'exp_different_shapes',
          jax.make_jaxpr(jnp.exp)(0.).jaxpr,
          jax.make_jaxpr(jnp.exp)(np.array([0.])).jaxpr,
          False,
          False
      ),
      (
          'exp_different_shapes_ignore',
          jax.make_jaxpr(jnp.exp)(0.).jaxpr,
          jax.make_jaxpr(jnp.exp)(np.array([0.])).jaxpr,
          True,
          True
      ),
      (
          'exp_vs_log',
          jax.make_jaxpr(jnp.exp)(0.).jaxpr,
          jax.make_jaxpr(jnp.log)(0.).jaxpr,
          False,
          False
      ),
      (
          'exp_float_vs_exp_int',
          jax.make_jaxpr(jnp.exp)(0.).jaxpr,
          jax.make_jaxpr(jnp.exp)(0).jaxpr,
          False,
          False
      ),
      (
          'plus_one',
          jax.make_jaxpr(lambda x: x+1)(0.).jaxpr,
          jax.make_jaxpr(lambda x: x+1)(0.).jaxpr,
          False,
          True
      ),
      (
          'plus_one_vs_plus_two',
          jax.make_jaxpr(lambda x: x+1)(0.).jaxpr,
          jax.make_jaxpr(lambda x: x+2)(0.).jaxpr,
          False,
          False
      ),
      (
          'softplus_p',
          jax.make_jaxpr(softplus_p.bind)(0.).jaxpr,
          jax.make_jaxpr(softplus_p.bind)(0.).jaxpr,
          False,
          True
      ),
      (
          'jax_nn_sigmoid',
          jax.make_jaxpr(jax.nn.sigmoid)(0.).jaxpr,
          jax.make_jaxpr(jax.nn.sigmoid)(0.).jaxpr,
          False,
          True
      ),
      (
          'sigmoid_vs_softplus',
          jax.make_jaxpr(jax.nn.sigmoid)(0.).jaxpr,
          jax.make_jaxpr(jax.nn.softplus)(0.).jaxpr,
          False,
          False
      ),
      (
          'jax_nn_softplus',
          jax.make_jaxpr(jax.nn.softplus)(0.).jaxpr,
          jax.make_jaxpr(jax.nn.softplus)(0.).jaxpr,
          False,
          True
      ),
      (
          'jax_nn_swish',
          jax.make_jaxpr(jax.nn.swish)(0.).jaxpr,
          jax.make_jaxpr(jax.nn.swish)(0.).jaxpr,
          False,
          True
      ),
  )
  def test_same_jaxpr_up_to_variable_renaming(self, j0, j1, ignore_shape,
                                              expected):
    actual = jaxpr_editor._same_jaxpr_up_to_variable_renaming(j0, j1,
                                                              ignore_shape)
    self.assertEqual(expected, actual)

  def assert_jaxpr_equiv(self, expected, actual):
    self.assertTrue(
        jaxpr_editor._same_jaxpr_up_to_variable_renaming(expected, actual))

  @parameterized.parameters(
      (jax.make_jaxpr(jnp.exp)(0.).jaxpr,),
      (jax.make_jaxpr(lambda x: 2*jnp.exp(x+1))(0.).jaxpr,),
      (jax.make_jaxpr(lambda x: (x * jnp.array([[1], [2]])).sum())(0.).jaxpr,),
  )
  def test_graph_conversion(self, jaxpr):
    graph = jaxpr_editor._jaxpr_to_graph(jaxpr)
    converted_jaxpr = jaxpr_editor._graph_to_jaxpr(graph)
    self.assert_jaxpr_equiv(jaxpr, converted_jaxpr)

  @parameterized.named_parameters(
      (
          'replace_exp_with_log',
          jax.make_jaxpr(jnp.exp)(0.).jaxpr,
          jax.make_jaxpr(jnp.log)(0.).jaxpr,
          jax.make_jaxpr(jnp.exp)(0.).jaxpr,
          jax.make_jaxpr(jnp.log)(0.).jaxpr,
      ),
      (
          'replace_exp_with_log_different_shapes',
          jax.make_jaxpr(jnp.exp)(0.).jaxpr,
          jax.make_jaxpr(jnp.log)(0.).jaxpr,
          jax.make_jaxpr(jnp.exp)(np.array([0.])).jaxpr,
          jax.make_jaxpr(jnp.log)(np.array([0.])).jaxpr,
      ),
      (
          'replace_exp_with_log_2',
          jax.make_jaxpr(jnp.exp)(0.).jaxpr,
          jax.make_jaxpr(jnp.log)(0.).jaxpr,
          jax.make_jaxpr(lambda x: 2*jnp.exp(x+1))(0.).jaxpr,
          jax.make_jaxpr(lambda x: 2*jnp.log(x+1))(0.).jaxpr,
      ),
      (
          'replace_exp_with_log_x_squared',
          jax.make_jaxpr(jnp.exp)(0.).jaxpr,
          jax.make_jaxpr(lambda x: jnp.log(x**2))(0.).jaxpr,
          jax.make_jaxpr(lambda x: 2*jnp.exp(x+1))(0.).jaxpr,
          jax.make_jaxpr(lambda x: 2*jnp.log((x+1)**2))(0.).jaxpr,
      ),
      (
          'replace_softplus_with_primitive',
          jax.make_jaxpr(jax.nn.softplus)(0.).jaxpr,
          jax.make_jaxpr(softplus_p.bind)(0.).jaxpr,
          jax.make_jaxpr(jax.nn.softplus)(0.).jaxpr,
          jax.make_jaxpr(softplus_p.bind)(0.).jaxpr,
      ),
      (
          'replace_softplus_with_primitive_different_shapes',
          jax.make_jaxpr(jax.nn.softplus)(0.).jaxpr,
          jax.make_jaxpr(softplus_p.bind)(0.).jaxpr,
          jax.make_jaxpr(jax.nn.softplus)(np.array([0.])).jaxpr,
          jax.make_jaxpr(softplus_p.bind)(np.array([0.])).jaxpr,
      ),
  )
  def test_jaxpr_replace(self, pattern, replacement, subject, expected):
    actual = jaxpr_editor.replace(pattern, replacement, subject)
    self.assert_jaxpr_equiv(expected, actual)


if __name__ == '__main__':
  absltest.main()
