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

import itertools
import math

from absl.testing import absltest
from absl.testing import parameterized
from autobound import elementwise_functions
from autobound import test_utils
import jax
import jax.numpy as jnp
import numpy as np


def get_jax_callable(function_id):
  jax_funs = {
      elementwise_functions.ABS.name: jnp.abs,
      elementwise_functions.EXP.name: jnp.exp,
      elementwise_functions.LOG.name: jnp.log,
      elementwise_functions.SIGMOID.name: jax.nn.sigmoid,
      elementwise_functions.SOFTPLUS.name: jax.nn.softplus,
      elementwise_functions.SWISH.name: jax.nn.swish,
  }
  if (function_id.name == elementwise_functions.ABS.name and
      function_id.derivative_order == 0):
    # Don't use jax.grad for the ABS function, because jax.grad(jnp.abs)(0.)
    # == 1, whereas our tests assume that every local minimum should have a
    # gradient of 0.
    return jnp.sign
  if function_id.name in jax_funs:
    f = jax_funs[function_id.name]
    for _ in range(function_id.derivative_order):
      f = jax.grad(f)
    return f
  else:
    raise NotImplementedError(function_id)


class TestCase(test_utils.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      (elementwise_functions.EXP,),
      (elementwise_functions.EXP.derivative_id(17),),
      (elementwise_functions.LOG,),
      (elementwise_functions.LOG.derivative_id(1),),
      (elementwise_functions.LOG.derivative_id(2),),
      (elementwise_functions.LOG.derivative_id(3),),
      (elementwise_functions.LOG.derivative_id(4),),
      (elementwise_functions.SIGMOID,),
      (elementwise_functions.SIGMOID.derivative_id(1),),
      (elementwise_functions.SIGMOID.derivative_id(2),),
      (elementwise_functions.SIGMOID.derivative_id(3),),
      (elementwise_functions.SIGMOID.derivative_id(4),),
      (elementwise_functions.SOFTPLUS,),
      (elementwise_functions.SOFTPLUS.derivative_id(1),),
      (elementwise_functions.SOFTPLUS.derivative_id(2),),
      (elementwise_functions.SOFTPLUS.derivative_id(3),),
      (elementwise_functions.SOFTPLUS.derivative_id(4),),
      (elementwise_functions.SOFTPLUS.derivative_id(5),),
      (elementwise_functions.SWISH,),
      (elementwise_functions.SWISH.derivative_id(1),),
      (elementwise_functions.SWISH.derivative_id(2),),
      (elementwise_functions.SWISH.derivative_id(3),),
      (elementwise_functions.SWISH.derivative_id(4),),
  )
  def test_get_function(self, function_id):
    actual = elementwise_functions.get_function(function_id, jnp)
    expected = get_jax_callable(function_id)
    for test_x in [-1000., -1., -.5, 0., .5, 1., 1000.]:
      if function_id.name == 'log' and test_x <= 0.:
        continue
      np.testing.assert_allclose(expected(test_x), actual(test_x))

  @parameterized.parameters(
      (elementwise_functions.EXP,
       elementwise_functions.FunctionData((), (),
                                          monotonically_increasing=True)),
      (elementwise_functions.EXP.derivative_id(17),
       elementwise_functions.FunctionData((), (),
                                          monotonically_increasing=True)),
  )
  def test_get_function_data(self, function_id, expected):
    actual = elementwise_functions.get_function_data(function_id)
    if expected is not None:
      self.assertEqual(expected, actual)
    self.sanity_check_function_data(function_id, actual)

  def test_all_function_data(self):
    for function_id, function_data in (
        elementwise_functions._FUNCTION_DATA.items()):
      self.sanity_check_function_data(function_id, function_data)

  @parameterized.parameters(
      (elementwise_functions.SIGMOID, (-1., 1.),
       (test_utils.sigmoid(-1.), test_utils.sigmoid(1.))),
      (elementwise_functions.SIGMOID, (-1e6, 1e6), (0., 1.)),
      (elementwise_functions.SIGMOID.derivative_id(1),
       (-2., 1.),
       (test_utils.sigmoid_derivative(1, -2.), test_utils.MAX_SIGMOID_DERIV)),
      (elementwise_functions.SIGMOID.derivative_id(1),
       (-2., -1.),
       (test_utils.sigmoid_derivative(1, -2.),
        test_utils.sigmoid_derivative(1, -1.))),
      (elementwise_functions.SIGMOID.derivative_id(1), (-1., 3.),
       (test_utils.sigmoid_derivative(1, 3.), test_utils.MAX_SIGMOID_DERIV)),
      (elementwise_functions.SIGMOID.derivative_id(1), (1., 3.),
       (test_utils.sigmoid_derivative(1, 3.),
        test_utils.sigmoid_derivative(1, 1.))),
      (elementwise_functions.SIGMOID.derivative_id(1), (-1e6, 1e6), (0., .25)),
      (elementwise_functions.SIGMOID.derivative_id(2), (-1e6, -4.),
       (0., test_utils.sigmoid_derivative(2, -4.))),
      (elementwise_functions.SIGMOID.derivative_id(2), (-4., -2.),
       (test_utils.sigmoid_derivative(2, -4.),
        test_utils.sigmoid_derivative(2, -2.))),
      (elementwise_functions.SIGMOID.derivative_id(2), (-2., -.5),
       (test_utils.sigmoid_derivative(2, -.5),
        test_utils.MAX_SIGMOID_SECOND_DERIV)),
      (elementwise_functions.SIGMOID.derivative_id(2), (-.5, .5),
       (test_utils.sigmoid_derivative(2, .5),
        test_utils.sigmoid_derivative(2, -.5))),
      (elementwise_functions.SIGMOID.derivative_id(2), (.5, 2.),
       (test_utils.MIN_SIGMOID_SECOND_DERIV,
        test_utils.sigmoid_derivative(2, .5))),
      (elementwise_functions.SIGMOID.derivative_id(2), (2., 5.),
       (test_utils.sigmoid_derivative(2, 2.),
        test_utils.sigmoid_derivative(2, 5.))),
      (elementwise_functions.SIGMOID.derivative_id(2), (5., 1e6),
       (test_utils.sigmoid_derivative(2, 5.), 0.)),
      (elementwise_functions.SIGMOID.derivative_id(3), (-1e6, -4.),
       (0., test_utils.sigmoid_derivative(3, -4.))),
      (elementwise_functions.SIGMOID.derivative_id(3), (-4., -3.),
       (test_utils.sigmoid_derivative(3, -4.),
        test_utils.sigmoid_derivative(3, -3.))),
      (elementwise_functions.SIGMOID.derivative_id(3), (-3., -1.),
       (test_utils.sigmoid_derivative(3, -1.),
        test_utils.MAX_SIGMOID_THIRD_DERIV)),
      (elementwise_functions.SIGMOID.derivative_id(3), (-1., .5),
       (test_utils.MIN_SIGMOID_THIRD_DERIV,
        test_utils.sigmoid_derivative(3, -1.))),
      (elementwise_functions.SIGMOID.derivative_id(3), (.5, 1.),
       (test_utils.sigmoid_derivative(3, .5),
        test_utils.sigmoid_derivative(3, 1.))),
      (elementwise_functions.SIGMOID.derivative_id(3), (1., 3.),
       (test_utils.sigmoid_derivative(3, 1.),
        test_utils.MAX_SIGMOID_THIRD_DERIV)),
      (elementwise_functions.SIGMOID.derivative_id(3), (3., 4.),
       (test_utils.sigmoid_derivative(3, 4.),
        test_utils.sigmoid_derivative(3, 3.))),
      (elementwise_functions.SIGMOID.derivative_id(3), (4., 1e6),
       (0., test_utils.sigmoid_derivative(3, 4.))),
      (elementwise_functions.SOFTPLUS.derivative_id(1), (-1., 1.),
       (test_utils.sigmoid(-1.), test_utils.sigmoid(1.))),
      (elementwise_functions.SWISH, (1., 2.),
       (test_utils.swish(1.), test_utils.swish(2.))),
  )
  def test_get_range(self, function_id, trust_region, expected):
    if callable(expected):
      expected = expected()
    for np_like in self.backends:
      actual = elementwise_functions.get_range(function_id, trust_region,
                                               np_like)
      self.assert_interval_equal(expected, actual)

  @parameterized.parameters(
      (
          elementwise_functions.EXP,
          0,
          3.14,
          (math.exp(3.14),)
      ),
      (
          elementwise_functions.EXP,
          2,
          3.14,
          (math.exp(3.14), math.exp(3.14), math.exp(3.14)/2)
      ),
      (
          elementwise_functions.LOG,
          0,
          3.14,
          (math.log(3.14),)
      ),
      (
          elementwise_functions.LOG,
          2,
          3.14,
          (math.log(3.14), 1/3.14, -.5/3.14**2)
      ),
  )
  def test_get_taylor_polynomial_coefficients(self, function_id, degree, x0,
                                              expected):
    for np_like in self.backends:
      actual = elementwise_functions.get_taylor_polynomial_coefficients(
          function_id, degree, x0, np_like)
      self.assert_enclosure_equal(expected, actual)

  @parameterized.parameters(
      # Function with no local minima.
      # Because there are no local minima, the function must either be
      # monotonically decreasing or monotonically increasing.
      (
          elementwise_functions.FunctionData((), (),
                                             monotonically_decreasing=True),
          ([-3., -2., -1., 0., 1., 2.], [-2., -1., 0., 1., 2., 3.]),
          (
              [True] * 6,
              [False] * 6,
          )
      ),
      (
          elementwise_functions.FunctionData((), (),
                                             monotonically_increasing=True),
          ([-3., -2., -1., 0., 1., 2.], [-2., -1., 0., 1., 2., 3.]),
          (
              [False] * 6,
              [True] * 6,
          )
      ),
      # Function with local minimum at -1 and local maximum at 1.
      (
          elementwise_functions.FunctionData((-1.,), (1.,)),
          ([-3., -2., -1., 0., 1., 2., -3.], [-2., -1., 0., 1., 2., 3., 3.]),
          (
              [True, True, False, False, True, True, False],
              [False, False, True, True, False, False, False],
          )
      ),
      # Function with local maximum at -1 and local minimum at 1.
      (
          elementwise_functions.FunctionData((1.,), (-1.,)),
          ([-3., -2., -1., 0., 1., 2., -3.], [-2., -1., 0., 1., 2., 3., 3.]),
          (
              [False, False, True, True, False, False, False],
              [True, True, False, False, True, True, False],
          )
      ),
  )
  def test_monotone_over(self, function_data, region, expected):
    expected_decreasing, expected_increasing = expected
    region = np.asarray(region)
    expected_decreasing = np.asarray(expected_decreasing)
    expected_increasing = np.asarray(expected_increasing)
    for np_like in self.backends:
      actual = function_data.monotone_over(region, np_like)
      self.assertIsInstance(actual, tuple)
      self.assertLen(actual, 2)
      actual_decreasing, actual_increasing = actual
      np.testing.assert_equal(actual_decreasing, expected_decreasing)
      np.testing.assert_equal(actual_increasing, expected_increasing)

  def sanity_check_function_data(self, function_id, function_data):
    # Make sure all local minima/maxima have 0 gradient.
    f = get_jax_callable(function_id)
    grad = jax.grad(f)
    for x in itertools.chain(function_data.local_minima,
                             function_data.local_maxima):
      np.testing.assert_allclose(grad(x), 0., rtol=0, atol=1e-12)
    # Make sure hessian is non-negative at local minima, and non-positive at
    # local maxima.
    hessian = jax.grad(grad)
    for x in function_data.local_minima:
      self.assertGreaterEqual(hessian(x), 0.)
    for x in function_data.local_maxima:
      self.assertLessEqual(hessian(x), 0.)

    if not function_data.local_minima and not function_data.local_maxima:
      self.assertTrue(function_data.monotonically_decreasing or
                      function_data.monotonically_increasing)

    if (function_data.monotonically_decreasing or
        function_data.monotonically_increasing):
      self.assertEqual((), function_data.local_minima)
      self.assertEqual((), function_data.local_maxima)

    if function_data.monotonically_increasing:
      self.assertLessEqual(f(1.), f(2.))
    if function_data.monotonically_decreasing:
      self.assertLessEqual(f(2.), f(1.))


if __name__ == '__main__':
  absltest.main()
