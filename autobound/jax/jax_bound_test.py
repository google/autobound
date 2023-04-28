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

import math

from absl.testing import absltest
from absl.testing import parameterized
from autobound import enclosure_arithmetic
from autobound import primitive_enclosures
from autobound import test_utils
from autobound.jax import jax_bound
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np


# Custom softplus primitive, for use in testing registration mechanism.
my_softplus_p = jax.core.Primitive('my_softplus')
my_softplus_p.def_abstract_eval(
    lambda x: jax.abstract_arrays.ShapedArray(x.shape, x.dtype))


def my_softplus(x):
  return my_softplus_p.bind(x)


class TestCase(parameterized.TestCase, test_utils.TestCase):

  @parameterized.parameters(
      (
          2,
          (np.zeros((1,)), np.ones((1,))),
          (np.ones((1,)), np.ones((1, 1))),
          (np.ones((1, 13, 7)),),
          {
              'dimension_numbers': (((0,), (0,)), ((), ())),
              'precision': None,
              'preferred_element_type': None
          },
          (np.ones((13, 7)), np.ones((13, 7, 1)))
      ),
      (
          2,
          (np.zeros((2,)), np.ones((2,))),
          (np.ones((3, 5)),),
          (np.ones((5, 7)), np.ones((5, 7, 2))),
          {
              'dimension_numbers': (((1,), (0,)), ((), ())),
              'precision': None,
              'preferred_element_type': None
          },
          (5*np.ones((3, 7)), 5*np.ones((3, 7, 2)))
      ),
      (
          2,
          (np.zeros((2,)), np.ones((2,))),
          (np.ones((5, 7)), np.ones((5, 7, 2))),
          (np.ones((7, 11)),),
          {
              'dimension_numbers': (((1,), (0,)), ((), ())),
              'precision': None,
              'preferred_element_type': None
          },
          (7*np.ones((5, 11)), 7*np.ones((5, 11, 2)))
      ),
      # TODO(mstreeter): test an example with batch dimensions.
  )
  def test_dot_general_pushforward_fun(self, max_degree, trust_region,
                                       lhs, rhs, params, expected):
    arithmetic = enclosure_arithmetic.TaylorEnclosureArithmetic(
        max_degree, trust_region, jnp)
    fun = jax_bound._dot_general_pushforward_fun(arithmetic)
    actual = fun(jax_bound._IntermediateEnclosure(enclosure=lhs),
                 jax_bound._IntermediateEnclosure(enclosure=rhs),
                 **params)
    self.assert_enclosure_equal(expected, actual)

  @parameterized.named_parameters(
      (
          'identity_0',
          lambda x: x,
          0,
          .1,
          (0., .25),
          False,
          ((0., .25),),
      ),
      (
          'identity_1',
          lambda x: x,
          1,
          3.14,
          (0, .25),
          False,
          (3.14, 1.),
      ),
      (
          'identity_2',
          lambda x: x,
          2,
          3.14,
          (0, .25),
          False,
          (3.14, 1.),
      ),
      (
          'addition',
          lambda x: 2 + x,
          1,
          3.14,
          (0, .25),
          False,
          (5.14, 1.),
      ),
      (
          'addition_prop',
          lambda x: 2 + x,
          1,
          3.14,
          (0, .25),
          True,
          (5.14, 1.),
      ),
      (
          'constant',
          lambda x: 2.,
          1,
          .5,
          (0, 1),
          False,
          (2.,),
      ),
      (
          'constant_prop_0',
          lambda x: jnp.exp(1),
          1,
          .5,
          (0, 1),
          True,
          (math.e,),
      ),
      (
          'constant_prop_1',
          lambda x: jnp.eye(3) @ (jnp.zeros((3,)) * x),
          1,
          .5,
          (0, 1),
          True,
          (np.zeros((3,)), np.zeros((3,))),
      ),
      (
          'multiplication',
          lambda x: 2*x,
          1,
          3.14,
          (0, .25),
          False,
          (6.28, 2.),
      ),
      (
          'abs',
          jnp.abs,
          1,
          np.array([0., 1., 1., -1., -1.]),
          (
              np.array([-1., 0., -1., -2., -2.]),
              np.array([2., 2., 2., 0., 1.])
          ),
          False,
          (np.array([0., 1., 1., 1., 1.]),
           (np.diag([-1., 1., 0., -1., -1.]), np.diag([1., 1., 1., -1., 0.])))
      ),
      (
          'exp',
          jnp.exp,
          2,
          1.,
          (0, 2),
          False,
          (math.e, math.e, (1., math.exp(2) - 2*math.e)),
      ),
      (
          'log_exp_noprop',
          lambda x: jnp.log(jnp.exp(x)),
          1,
          1.,
          (.5, 2),
          False,
          # Degree-1 enclosure for exp(x) at 1 over [.5, 2] is:
          #   (e, (2*(e-e**.5), e**2 - e)).
          # Enclosing this degree-1 enclosure by an interval, using the fact
          # that x-x0 in [-.5, 1], gives:
          #   (e - .5*(e**2 - e), e**2)
          # Degree-1 enclosure for log(y) at e over [e - .5*(e**2 - e), e**2]
          # is:
          #   (1, [1/(e**2 - e), 2*(1 - log(1.5*e - .5*e**2)) / (e**2 - e)])
          # Composing the two degree-1 enclosures gives the following expected
          # enclosure:
          (1., (2*(math.e-math.e**.5)/(math.e**2-math.e),
                2*(1 - math.log(1.5*math.e - .5*math.e**2)))),
          # Expected value is approximately (1, (0.45797998, 3.91999056)).
      ),
      (
          'log_exp_prop',
          lambda x: jnp.log(jnp.exp(x)),
          1,
          1.,
          (0.5, 2),
          True,
          # Degree-0 enclosure for exp(x) at 1 over [.5, 2] is [e**.5, e**2]
          # Degree-1 enclosure for log(y) at e over [e**.5, e**2] is
          #   (1, (1/(e**2 - e) .5/(e - e**.5)))
          # Degree-1 enclosure for exp(x) at 1 over [.5, 2] is:
          #   (e, (2*(e-e**.5), e**2 - e)).
          # Composing the two degree-1 enclosures gives the following expected
          # enclosure:
          (1., (2*(math.e-math.e**.5)/(math.e**2-math.e),
                .5*(math.e**2-math.e)/(math.e-math.e**.5))),
          # Expected value is approximately (1, (0.45797998, 2.18350155)).
      ),
      (
          'log',
          jnp.log,
          2,
          2.,
          (1, 3),
          False,
          (np.log(2), .5, (-np.log(2)+.5, np.log(3)-np.log(2)-.5))
      ),
      (
          'eye',
          lambda x: jnp.eye(3),
          0,
          3.14,
          (0, .25),
          False,
          (np.eye(3),)
      ),
      (
          'matmul_a',
          lambda x: jnp.matmul(jnp.eye(3), x*jnp.ones((3,))),
          1,
          lambda: jnp.array(1.),
          lambda: (jnp.array(0.), jnp.array(2.)),
          False,
          (np.ones((3,)), np.ones((3,)))
      ),
      (
          'matmul_b',
          lambda x: jnp.matmul(jnp.eye(3), x),
          1,
          lambda: jnp.ones((3,)),
          lambda: (jnp.zeros((3,)), 2*jnp.ones((3,))),
          False,
          (np.ones((3,)), np.eye(3))
      ),
      (
          'tensordot',
          lambda x: jnp.tensordot(x, jnp.array([[6., 8.]]), axes=1),
          1,
          np.array([0.]),
          (np.array([0.]), np.array([1.])),
          False,
          (np.array([0., 0.]), np.array([[6.], [8.]])),
      ),
      (
          'negative',
          jnp.negative,
          0,
          3.14,
          (0., 5.),
          False,
          ((-5., 0.),)
      ),
      (
          'minus',
          lambda x: x - 1,
          0,
          3.14,
          (0., 5.),
          False,
          ((-1., 4.),)
      ),
      (
          'broadcast',
          lambda x: jax.lax.broadcast(x, [2]),
          0,
          3.14,
          (0., 5.),
          False,
          (([0., 0.], [5., 5.]),)
      ),
      (
          'square_a',
          lambda x: x**2,
          0,
          0,
          (-.5, .5),
          False,
          ((0., .25),),
      ),
      (
          'square_b',
          lambda x: x**2,
          2,
          0.,
          (0, 1),
          False,
          (0., 0., 1.),
      ),
      (
          'square_c',
          lambda x: x**2,
          2,
          0.5,
          (0, 1),
          False,
          (0.25, 1., 1.),
      ),
      (
          'square_b_prop',
          lambda x: x**2,
          2,
          0.5,
          (0, 1),
          True,
          (0.25, 1., 1.),
      ),
      (
          'sqrt',
          lambda x: x**.5,
          2,
          2.,
          (1, 3),
          False,
          (2**.5, .5 / 2**.5,
           (1 - (2**.5 - .5/2**.5), 3**.5 - (2**.5 + .5/2**.5)))
      ),
      (
          'multiply_b',
          lambda x: x * jnp.array([[1], [2]]),
          2,
          0.5,
          (0, 2),
          False,
          (np.array([[0.5], [1.]]), np.array([[1.], [2.]]))
      ),
      (
          'reshape',
          lambda x: (x * jnp.array([[1], [2]])).reshape(-1),
          2,
          0.5,
          (0, 2),
          False,
          (np.array([0.5, 1.]), np.array([1., 2.]))
      ),
      (
          'sum',
          lambda x: (x * jnp.array([[1], [2]])).sum(),
          2,
          0.5,
          (0, 2),
          False,
          (1.5, 3.)
      ),
      (
          'transpose',
          lambda x: (x * jnp.array([[1], [2]])).transpose(),
          2,
          0.5,
          (0, 2),
          False,
          (np.array([[0.5, 1.]]), np.array([[1., 2.]]))
      ),
      (
          'squeeze',
          lambda x: (x * jnp.ones((4, 1, 2))).squeeze([1]),
          2,
          0.5,
          (0, 2),
          False,
          (.5 * np.ones((4, 2)), np.ones((4, 2)))
      ),
      (
          'avg_pool_a',
          lambda x: nn.avg_pool(x*jnp.array([[[1], [2], [3], [4]]]), (2,)),
          2,
          0.5,
          (0, 2),
          False,
          (.5 * np.array([[[1.5], [2.5], [3.5]]]),
           np.array([[[1.5], [2.5], [3.5]]]))
      ),
      (
          'avg_pool_b',
          lambda x: nn.avg_pool((x**2)*jnp.array([[[1], [2], [3], [4]]]), (2,)),
          2,
          3.,
          (0, 5),
          False,
          (
              9 * np.array([[[1.5], [2.5], [3.5]]]),
              6 * np.array([[[1.5], [2.5], [3.5]]]),
              np.array([[[1.5], [2.5], [3.5]]])
          )
      ),
      (
          'my_softplus',
          my_softplus,
          2,
          0.,
          (-1., 1.),
          False,
          (math.log(2), .5, (math.log(1+math.exp(1)) - math.log(2) - .5, .125)),
      ),
      (
          'conv_general_dilated_a',
          # This returns x**2 * ones((1,1,1,1)).
          # pylint: disable=g-long-lambda
          lambda x: jax.lax.conv_general_dilated(x*jnp.ones((1, 1, 1, 1)),
                                                 x*jnp.ones((1, 1, 1, 1)),
                                                 [1, 1], 'VALID'),
          2,
          3.,
          (0, 5),
          False,
          (
              9.*np.ones((1, 1, 1, 1)),
              6.*np.ones((1, 1, 1, 1)),
              np.ones((1, 1, 1, 1))
          )
      ),
      (
          'conv_general_dilated_b',
          # This returns 5 * x**2 * ones((1,1,1,1)).
          # pylint: disable=g-long-lambda
          lambda x: jax.lax.conv_general_dilated(x*jnp.ones((1, 5, 1, 1)),
                                                 x*jnp.ones((1, 5, 1, 1)),
                                                 [1, 1], 'VALID'),
          2,
          3.,
          (0, 5),
          False,
          (
              9.*5*np.ones((1, 1, 1, 1)),
              6.*5*np.ones((1, 1, 1, 1)),
              5*np.ones((1, 1, 1, 1))
          )
      ),
      (
          'conv_general_dilated_c',
          # This returns 2 * x**2 * ones((2,3,5,6)).
          # pylint: disable=g-long-lambda
          lambda x: jax.lax.conv_general_dilated(x*jnp.ones((2, 1, 5, 7)),
                                                 x*jnp.ones((3, 1, 1, 2)),
                                                 [1, 1], 'VALID'),
          2,
          3.,
          (0, 5),
          False,
          (
              9.*2*np.ones((2, 3, 5, 6)),
              6.*2*np.ones((2, 3, 5, 6)),
              2*np.ones((2, 3, 5, 6))
          )
      ),
      # TODO(mstreeter): test convolutions where the input is not a scalar.
      (
          'jax_nn_sigmoid',
          jax.nn.sigmoid,
          0,
          0.,
          (-1., 1.),
          False,
          ((test_utils.sigmoid(-1.), test_utils.sigmoid(1.)),)
      ),
      (
          'jax_nn_softplus',
          jax.nn.softplus,
          0,
          0.,
          (-1., 1.),
          False,
          ((test_utils.softplus(-1.), test_utils.softplus(1.)),)
      ),
      (
          'jax_nn_swish',
          jax.nn.swish,
          0,
          2.,
          (1., 3.),
          False,
          ((test_utils.swish(1), test_utils.swish(3)),)
      ),
      (
          'jax_nn_sigmoid_ndarray',
          jax.nn.sigmoid,
          0,
          np.array([0., 1.]),
          (np.array([-1., -2.]), np.array([1., 2.])),
          False,
          (
              (
                  np.array([test_utils.sigmoid(-1.), test_utils.sigmoid(-2.)]),
                  np.array([test_utils.sigmoid(1.), test_utils.sigmoid(2.)]),
              ),
          )
      ),
      (
          'jax_nn_softplus_ndarray',
          jax.nn.softplus,
          0,
          np.array([0., 1.]),
          (np.array([-1., -2.]), np.array([1., 2.])),
          False,
          (
              (
                  np.array([test_utils.softplus(-1.),
                            test_utils.softplus(-2.)]),
                  np.array([test_utils.softplus(1.), test_utils.softplus(2.)])
              ),
          )
      ),
      (
          'jax_nn_swish_ndarray',
          jax.nn.swish,
          0,
          np.array([2., 3.]),
          (np.array([1., 2.]), np.array([3., 4.])),
          False,
          (
              (
                  np.array([test_utils.swish(1.), test_utils.swish(2.)]),
                  np.array([test_utils.swish(3.), test_utils.swish(4.)])
              ),
          )
      ),
      (
          'multiple_unused_output_variables',
          lambda x: jax.grad(lambda x: x**2)(x) + jax.grad(lambda x: x**2)(x),
          0,
          0.,
          (0., 1.),
          False,
          ((0., 4.),)
      ),
  )
  def test_taylor_bounds(
      self, f, max_degree, test_x0, test_trust_region, propagate_trust_regions,
      expected_coefficients):
    if callable(test_x0):
      test_x0 = test_x0()
    if callable(test_trust_region):
      test_trust_region = test_trust_region()
    g = jax_bound.taylor_bounds(f, max_degree, propagate_trust_regions)
    actual_coefficients = g(test_x0, test_trust_region).coefficients
    self.assert_enclosure_equal(expected_coefficients, actual_coefficients,
                                rtol=1e-6)

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    jax_bound.register_elementwise_primitive(
        my_softplus_p,
        primitive_enclosures.softplus_enclosure
    )


if __name__ == '__main__':
  absltest.main()
