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

import functools
import math

from absl.testing import absltest
from absl.testing import parameterized
from autobound import polynomials
from autobound import primitive_enclosures
from autobound import test_utils
import jax
import jax.numpy as jnp
import numpy as np


class TestCase(parameterized.TestCase, test_utils.TestCase):

  def assert_is_taylor_enclosure(
      self,
      f,
      enclosure,
      x0,
      trust_region,
      lower_tight_at,
      upper_tight_at,
      num_test_points=101,
      eps=1e-12,
      **kwargs):
    for p in np.linspace(0, 1, num_test_points):
      x = trust_region[0]*(1-p) + trust_region[1]*p
      value = f(x)
      result = polynomials.eval_elementwise_taylor_enclosure(
          enclosure, x-x0, np)
      if isinstance(result, tuple):
        lower_bound, upper_bound = result
      else:
        lower_bound = result
        upper_bound = result
      np.testing.assert_array_less(lower_bound - eps, value)
      np.testing.assert_array_less(value, upper_bound + eps)

    self.assert_is_taylor_series(f, enclosure[:-1], x0, **kwargs)

    for x in lower_tight_at:
      self.assert_lower_bound_tight_at(f, enclosure, x0, x)
    for x in upper_tight_at:
      self.assert_upper_bound_tight_at(f, enclosure, x0, x)

  def assert_is_taylor_series(self, f, series, x0, **kwargs):
    expected_series = self.expected_taylor_coefficients(f, x0, len(series) - 1)
    assert len(expected_series) == len(series)
    for expected, actual in zip(expected_series, series):
      self.assert_allclose_strict(expected, actual)

  def assert_lower_bound_tight_at(self, f, enclosure, x0, x):
    lower_bound, _ = polynomials.eval_elementwise_taylor_enclosure(
        enclosure, x-x0, np)
    value = f(x)
    np.testing.assert_allclose(lower_bound, value)

  def assert_upper_bound_tight_at(self, f, enclosure, x0, x):
    _, upper_bound = polynomials.eval_elementwise_taylor_enclosure(
        enclosure, x-x0, np)
    value = f(x)
    np.testing.assert_allclose(upper_bound, value)

  @classmethod
  def expected_derivative(cls, f, x, order):
    is_scalar_valued = np.asarray(f(x)).ndim == 0
    if is_scalar_valued:
      g = f
      for _ in range(order):
        g = jax.grad(g)
      return g(x)
    else:
      raise NotImplementedError()

  @classmethod
  def expected_taylor_coefficients(cls, f, x0, degree):
    """Return Taylor polynomial coefficients for elementwise function."""
    is_scalar_valued = np.asarray(f(x0)).ndim == 0
    if is_scalar_valued:
      # Compute expected Taylor polynomial coefficients using jax.grad.
      expected_coefficients = []
      ith_deriv = f
      i_factorial = 1
      for i in range(degree + 1):
        if i > 0:
          i_factorial *= i
        expected_coefficients.append(ith_deriv(x0) / i_factorial)
        ith_deriv = jax.grad(ith_deriv)
      return expected_coefficients
    else:
      expected_coefficients = []
      for i in range(degree + 1):
        if i == 0:
          expected_coefficients.append(f(x0))
        elif i == 1:
          # f is assumed to be elementwise.
          grad_diag = jnp.diag(jax.jacobian(f)(x0))
          expected_coefficients.append(grad_diag)
        else:
          raise NotImplementedError(i)
      return expected_coefficients

  @parameterized.parameters(
      # Test p==0.  0th deriv is 1, higher derivs are 0.
      (1, 0, 0, 1),
      (-1, 0, 0, 1),
      (1, 0, 1, 0),
      (-1, 0, 1, 0),
      (1, 2, 3, 0),
      (-1, 2, 3, 0),
      # Test p==1.
      (1, 1, 0, 1),
      (1, 1, 1, 1),
      (1, 1, 2, 0),
      (-1, 1, 0, -1),
      (-1, 1, 1, 1),
      (-1, 1, 2, 0),
      # Test p==2.
      (1, 2, 0, 1),
      (1, 2, 1, 1),
      (1, 2, 2, 1),
      (1, 2, 3, 0),
      (-1, 2, 0, 1),
      (-1, 2, 1, -1),
      (-1, 2, 2, 1),
      (-1, 2, 3, 0),
  )
  def test_pow_kth_deriv_sign(self, x_sign, p, k, expected):
    actual = primitive_enclosures._pow_kth_deriv_sign(x_sign, p, k)
    self.assertEqual(expected, actual)

  @parameterized.named_parameters(
      (
          'abs_1',
          jnp.abs,
          primitive_enclosures.abs_enclosure,
          0.,
          (-1., 2.),
          1,
          (0., (-1., 1.))
      ),
      (
          'abs_2',
          jnp.abs,
          primitive_enclosures.abs_enclosure,
          1.,
          (0., 2.),
          1,
          (1., (1., 1.))
      ),
      (
          'abs_3',
          jnp.abs,
          primitive_enclosures.abs_enclosure,
          1.,
          (-1., 2.),
          1,
          (1., (0., 1.))
      ),
      (
          'abs_4',
          jnp.abs,
          primitive_enclosures.abs_enclosure,
          -1.,
          (-2., 0.),
          1,
          (1., (-1., -1.))
      ),
      (
          'abs_5',
          jnp.abs,
          primitive_enclosures.abs_enclosure,
          -1.,
          (-2., 1.),
          1,
          (1., (-1., 0.))
      ),
      (
          'abs_ndarray',
          jnp.abs,
          primitive_enclosures.abs_enclosure,
          # Tests abs_1 through abs_5, as a single ndarray.
          np.array([0., 1., 1., -1., -1.]),
          (
              np.array([-1., 0., -1., -2., -2.]),
              np.array([2., 2., 2., 0., 1.])
          ),
          1,
          (np.array([0., 1., 1., 1., 1.]),
           (np.array([-1., 1., 0., -1., -1.]), np.array([1., 1., 1., -1., 0.])))
      ),
      (
          'exp_degree_0',
          jnp.exp,
          primitive_enclosures.exp_enclosure,
          1,
          (0, 2),
          0,
          ((1., math.e**2),),
          [0],
          [2]
      ),
      (
          'exp_degree_1',
          jnp.exp,
          primitive_enclosures.exp_enclosure,
          1,
          (0, 2),
          1,
          (math.e, (math.e - 1, math.e**2 - math.e)),
          [],
          [0, 2]
      ),
      (
          'exp_degree_2',
          jnp.exp,
          primitive_enclosures.exp_enclosure,
          1,
          (0, 2),
          2,
          # T_1(x; exp, 1) = e + e(x-1)
          # T_1(0; exp, 1) = 0
          # R_1(0; exp, 1) = 1
          (math.e, math.e, (1., math.e**2 - 2*math.e)),
          [0],
          [2]
      ),
      (
          'exp_degree_3',
          jnp.exp,
          primitive_enclosures.exp_enclosure,
          1,
          (0, 2),
          3,
          # T_2(x; exp, 1) = e + e(x-1) + e(x-1)**2 / 2
          # T_2(0; exp, 1) = e/2
          # R_2(0, exp, 1) = 1-e/2
          # T_2(2; exp, 1) = 2.5 e
          # R_2(2, exp, 1) = e**2 - 2.5 e
          (math.e, math.e, math.e / 2, (math.e/2 - 1, math.e**2 - 2.5*math.e)),
          [],
          [0, 2]
      ),
      (
          'exp_degree_4',
          jnp.exp,
          primitive_enclosures.exp_enclosure,
          1,
          (0, 2),
          4,
          # T_3(x; exp, 1) = e + e(x-1) + e(x-1)**2 / 2 + e(x-1)**3 / 6
          # T_3(0; exp, 1) = e/3
          # R_3(0, exp, 1) = 1-e/3
          # T_3(2; exp, 1) = (2 + 2/3) e
          # R_2(2, exp, 1) = e**2 - (2 + 2/3) e
          (math.e, math.e, math.e / 2, math.e / 6,
           (1 - math.e/3, math.e**2 - (2+2/3)*math.e)),
          [0],
          [2]
      ),
      (
          'exp_ndarray',
          jnp.exp,
          primitive_enclosures.exp_enclosure,
          np.array([1, -2, 3]),
          (np.array([0, -2, 2]), np.array([2, 1, 3])),
          2,
          (
              np.array([math.e, math.e**-2, math.e**3]),
              np.array([math.e, math.e**-2, math.e**3]),
              (
                  np.array([1,
                            math.e**-2 / 2,
                            math.e**2]),
                  np.array([math.e**2-2*math.e,
                            (math.e-4*math.e**-2)/9,
                            math.e**3 / 2])
              )
          ),
      ),
      (
          'log_scalar',
          jnp.log,
          primitive_enclosures.log_enclosure,
          2,
          (1, 3),
          2,
          (np.log(2), .5, (-np.log(2)+.5, np.log(3)-np.log(2)-.5)),
          [1],
          [3]
      ),
      (
          'log_ndarray',
          jnp.log,
          primitive_enclosures.log_enclosure,
          np.array([2, 3, 4]),
          (np.array([1, 3, 3]), np.array([3, 4, 4])),
          2,
          (
              np.array([math.log(2), math.log(3), math.log(4)]),
              np.array([.5, 1/3, 1/4]),
              (
                  np.array([-math.log(2)+.5,
                            -.5/3**2,
                            math.log(3) - math.log(4) + 1/4]),
                  np.array([np.log(3)-np.log(2)-.5,
                            math.log(4) - math.log(3) - 1/3,
                            -.5/4**2])
              ),
          )
      ),
      (
          'sigmoid_degree_0',
          jax.nn.sigmoid,
          primitive_enclosures.sigmoid_enclosure,
          0.,
          (-1, 1),
          0,
          ((test_utils.sigmoid(-1.), test_utils.sigmoid(1.)),),
      ),
      (
          'sigmoid_degree_1',
          jax.nn.sigmoid,
          primitive_enclosures.sigmoid_enclosure,
          0.,
          (-1, 1),
          1,
          (test_utils.sigmoid(0.),
           (test_utils.sigmoid_derivative(1, -1.),
            test_utils.MAX_SIGMOID_DERIV),),
      ),
      (
          'sigmoid_degree_2',
          jax.nn.sigmoid,
          primitive_enclosures.sigmoid_enclosure,
          0.,
          (-1, 1),
          2,
          (
              test_utils.sigmoid(0.),
              test_utils.sigmoid_derivative(1, 0.),
              (
                  test_utils.sigmoid(1.) - (
                      test_utils.sigmoid(0.) +
                      test_utils.sigmoid_derivative(1, 0.) +
                      test_utils.sigmoid_derivative(2, 0.)/2),
                  test_utils.sigmoid(-1.) - (
                      test_utils.sigmoid(0.) -
                      test_utils.sigmoid_derivative(1, 0.) -
                      test_utils.sigmoid_derivative(2, 0.)/2),
              ),
          ),
      ),
      (
          'sigmoid_degree_3',
          jax.nn.sigmoid,
          primitive_enclosures.sigmoid_enclosure,
          0.,
          (-1, 1),
          3,
          (test_utils.sigmoid(0.), test_utils.sigmoid_derivative(1, 0.),
           test_utils.sigmoid_derivative(2, 0.)/2,
           (test_utils.MIN_SIGMOID_THIRD_DERIV/6,
            test_utils.sigmoid_derivative(3, -1.)/6),),
      ),
      (
          'softplus_degree_0',
          jax.nn.softplus,
          primitive_enclosures.softplus_enclosure,
          0.,
          (-1, 1),
          0,
          ((test_utils.softplus(-1), test_utils.softplus(1)),),
      ),
      (
          'softplus_degree_1',
          jax.nn.softplus,
          primitive_enclosures.softplus_enclosure,
          0.,
          (-1, 1),
          1,
          # The 1st derivative (sigmoid) is monotonically increasing, so we
          # should get the sharp degree-1 enclosure here.
          (test_utils.softplus(0.),
           (test_utils.softplus(0.) - test_utils.softplus(-1.),
            test_utils.softplus(1.) - test_utils.softplus(0.)),),
      ),
      (
          'softplus_degree_2',
          jax.nn.softplus,
          primitive_enclosures.softplus_enclosure,
          0.,
          (-1, 1),
          2,
          (math.log(2), .5, (test_utils.softplus(1) - math.log(2) - .5, .125)),
      ),
      (
          'softplus_degree_3',
          jax.nn.softplus,
          primitive_enclosures.softplus_enclosure,
          0.,
          (-1, 1),
          3,
          (
              test_utils.softplus(0.),
              test_utils.sigmoid(0.),
              test_utils.sigmoid_derivative(1, 0.) / 2,
              (
                  test_utils.softplus(1.) - (
                      test_utils.softplus(0.) +
                      test_utils.softplus_derivative(1, 0.) +
                      test_utils.softplus_derivative(2, 0.)/2),
                  (test_utils.softplus(-1.) - (
                      test_utils.softplus(0.) -
                      test_utils.softplus_derivative(1, 0.) +
                      test_utils.softplus_derivative(2, 0.)/2)) / -1.
              ),
          ),
      ),
      (
          'softplus_degree_4',
          jax.nn.softplus,
          primitive_enclosures.softplus_enclosure,
          0.,
          (-1, 1),
          4,
          (
              test_utils.softplus(0.),
              test_utils.sigmoid(0.),
              test_utils.sigmoid_derivative(1, 0.)/2,
              test_utils.sigmoid_derivative(2, 0.)/6,
              (test_utils.MIN_SIGMOID_THIRD_DERIV/24,
              test_utils.sigmoid_derivative(3, -1.)/24),
          ),
      ),
      (
          'softplus_ndarray',
          lambda x: jnp.log1p(jnp.exp(x)),
          primitive_enclosures.softplus_enclosure,
          np.array([0., 0., 0.]),
          (np.array([-1., 0., -2.]), np.array([1., 2., 0.])),
          2,
          (
              np.array([math.log(2), math.log(2), math.log(2)]),
              np.array([.5, .5, .5]),
              (
                  np.array([
                      math.log(1+math.exp(1)) - math.log(2) - .5,
                      (math.log(1+math.exp(2)) - math.log(2) - 1)/4,
                      (math.log(1+math.exp(2)) - math.log(2) - 1)/4
                  ]),
                  np.array([.125, .125, .125]),
              )
          ),
      ),
      (
          'reciprocal',
          lambda x: 1 / x,
          functools.partial(primitive_enclosures.pow_enclosure, -1),
          2,
          (1, 3),
          2,
          # R1(x) = 1/x - .5 + .25(x-2)
          # R1(1) = .25
          # R1(3) = 1/3 - 1/2 + 1/4 = 1/12.
          (.5, -.25, (1/12, .25)),
      ),
      (
          'square_degree_1',
          lambda x: x**2,
          functools.partial(primitive_enclosures.pow_enclosure, 2),
          3.,
          (-1, 5),
          1,
          # R_0(x; x-->x**2, 3) = x**2 - 9
          # R_0(-1; x-->x**2, 3) / (-1-3) = (1 - 9) / -4 = 2.
          # R_0(5; x-->x**2, 3) / (5-3) = (25 - 9) / 2 = 8.
          (9., (2., 8.)),
      ),
      (
          'square_degree_2',
          lambda x: x**2,
          functools.partial(primitive_enclosures.pow_enclosure, 2),
          3.,
          (-1, 5),
          2,
          (9., 6., (1., 1.)),
      ),
      (
          'sqrt',
          lambda x: x**.5,
          functools.partial(primitive_enclosures.pow_enclosure, .5),
          2.,
          (1., 3.),
          2,
          # T_1(x; sqrt, 2) = sqrt(2) + (x-2) * .5 / sqrt(2)
          # R_1(x; sqrt, 2) = sqrt(x) - sqrt(2) - (x-2) * .5 / sqrt(x)
          # R_1(1; sqrt, 2) / (1 - 2)**2 = 1 - (sqrt(2) - .5 / sqrt(x))
          # R_1(3; sqrt, 2) / (3 - 2)**2 = sqrt(3) - (sqrt(2) + .5 / sqrt(2))
          (2**.5, .5 / 2**.5,
           (1 - (2**.5 - .5/2**.5), 3**.5 - (2**.5 + .5/2**.5))),
      ),
      (
          'sqrt_0_left_endpoint',
          lambda x: x**.5,
          functools.partial(primitive_enclosures.pow_enclosure, .5),
          2.,
          (0., 3.),
          2,
          # As in previous test, except left endpoint is now:
          # R_1(0; sqrt, 2) / (0 - 2)**2 = [-sqrt(2) + 1 / sqrt(2)] / 4.
          (2**.5, .5 / 2**.5,
           ((-2**.5 + 1/2**.5) / 4, 3**.5 - (2**.5 + .5/2**.5))),
      ),
      (
          'cbrt',
          lambda x: x**(1/3),
          functools.partial(primitive_enclosures.pow_enclosure, 1/3),
          2.5,
          (2., 3.),
          0,
          ((2**(1/3), 3**(1/3)),),
      ),
      # TODO(mstreeter): test enclosure for sqrt(x) when trust region includes
      # negative values.
      (
          'swish_degree_0',
          jax.nn.swish,
          primitive_enclosures.swish_enclosure,
          2.,
          # The swish function is monotonically increasing over [1, 3].
          (1., 3.),
          0,
          ((test_utils.swish(1.), test_utils.swish(3.)),),
          [1.],
          [3.]
      ),
      (
          'swish_degree_1',
          jax.nn.swish,
          primitive_enclosures.swish_enclosure,
          .5,
          # Derivative of swish is monotonically increasing over [-1, 1].
          (-1., 1.),
          1,
          (
              test_utils.swish(.5),
              (
                  (test_utils.swish(-1.) - test_utils.swish(.5)) / -1.5,
                  (test_utils.swish(1.) - test_utils.swish(.5)) / .5
              ),
          ),
      ),
      (
          'swish_degree_2',
          jax.nn.swish,
          primitive_enclosures.swish_enclosure,
          .5,
          # Second derivative of swish is has a local maximum at x=0.
          (-1., 1.),
          2,
          (
              test_utils.swish(.5),
              test_utils.swish_derivative(1, .5),
              (test_utils.swish_derivative(2, -1.) / 2,
               test_utils.swish_derivative(2, 0.) / 2),
          ),
      ),
      (
          'swish_degree_3',
          jax.nn.swish,
          primitive_enclosures.swish_enclosure,
          .5,
          # Third derivative of swish is monotonically decreasing over [-1, 1].
          (-1., 1.),
          3,
          (
              test_utils.swish(.5),
              test_utils.swish_derivative(1, .5),
              test_utils.swish_derivative(2, .5) / 2,
              (
                  (test_utils.swish(1.) -
                   (test_utils.swish(.5) +
                    .5*test_utils.swish_derivative(1, .5) +
                    (.5**2)*test_utils.swish_derivative(2, .5)/2)) / .5**3,
                  (test_utils.swish(-1.) -
                   (test_utils.swish(.5) +
                    -1.5*test_utils.swish_derivative(1, .5) +
                    (1.5**2)*test_utils.swish_derivative(2, .5)/2)) / (-1.5)**3
              ),
          ),
      ),
      (
          'swish_degree_4',
          jax.nn.swish,
          primitive_enclosures.swish_enclosure,
          .5,
          # Fourth derivative of swish is even symmetric, with a local minimum
          # at x=0.
          (-1., 1.),
          4,
          (
              test_utils.swish(.5),
              test_utils.swish_derivative(1, .5),
              test_utils.swish_derivative(2, .5) / 2,
              test_utils.swish_derivative(3, .5) / 6,
              (test_utils.swish_derivative(4, 0.) / 24,
               test_utils.swish_derivative(4, -1.) / 24),
          ),
      ),
  )
  def test_enclosure_generator(self, f, f_enclosure_generator, x0, trust_region,
                               degree, expected, lower_tight_at=(),
                               upper_tight_at=()):
    if callable(expected):
      expected = expected()
    # Making x0 float64 is necessary so that Taylor enclosure coefficients are
    # computed precisely enough for tests to pass.
    x0 = np.array(x0, dtype=np.float64)
    self.assert_is_taylor_enclosure(f, expected, x0, trust_region,
                                    lower_tight_at, upper_tight_at)
    for np_like in self.backends:
      actual = f_enclosure_generator(x0, trust_region, degree, np_like)
      self.assert_enclosure_equal(expected, actual, rtol=1e-6)
      self.assert_is_taylor_enclosure(f, actual, x0, trust_region,
                                      lower_tight_at, upper_tight_at)

  @parameterized.parameters(
      # Test that quadratic enclosure is tight for quadratics.
      (
          3,
          2,
          (0, 5),
          lambda x: x**2,
          (9, 6, 1),
          True,
          (9, 6, (1., 1.))
      ),
      # Non-quadratic, monotonically-increasing Hessian.
      (
          .5,
          2,
          (0, 1),
          lambda x: x**2 + x**3,
          (.375, 1.75, 2.5),
          True,
          (.375, 1.75, (2., 3.))
      ),
      # Non-quadratic, monotonically-decreasing Hessian.
      (
          .5,
          2,
          (0, 1),
          lambda x: x**2 - x**3,
          (.125, .25, -.5),
          False,
          (.125, .25, (-1., 0.)),
      ),
      # When trust region is (x0, x0), quadratic coefficient should be
      # Hessian/2.
      (
          .5,
          2,
          (.5, .5),
          lambda x: x**2 - x**3,
          (.125, .25, .25),
          False,
          (.125, .25, (.25, .25))
      ),
      # Trust region (x0, x0+eps).
      (
          .5,
          2,
          (.5, .5+1e-12),
          lambda x: x**2 - x**3,
          (.125, .25, .25),
          False,
          (.125, .25, (.25, .25))
      ),
      # Trust region (x0-eps, x0).
      (
          .5,
          2,
          (.5, .5+1e-12),
          lambda x: x**2 - x**3,
          (.125, .25, .25),
          False,
          (.125, .25, (.25, .25))
      ),
      # Make sure ndarrays are handled as expected.
      (
          np.array([.5, 1]),
          2,
          (np.array([0, 0]), np.array([1, 2])),
          lambda x: x**2 + x**3,
          (
              np.array([.375, 2]),
              np.array([1.75, 5]),
              np.array([2.5, 4]),
          ),
          True,
          (
              np.array([.375, 2.]),
              np.array([1.75, 5.]),
              (
                  np.array([2., 3.]),
                  np.array([3., 5.]),
              )
          ),
      ),
  )
  def test_sharp_enclosure_monotonic_derivative(
      self, x0, degree, trust_region, sigma, taylor_coefficients_at_x0,
      increasing, expected):
    for np_like in self.backends:
      actual = primitive_enclosures.sharp_enclosure_monotonic_derivative(
          x0, degree, trust_region, sigma, taylor_coefficients_at_x0,
          increasing, np_like)
      self.assert_enclosure_equal(expected, actual)

  @parameterized.parameters(
      # f(x) = 0.
      (
          0.,
          (-1., 1.),
          lambda _: 0.,
          (0., 0., 0.),
          (0., 0., (0., 0.))
      ),
      # f(x) = x**2.
      (
          3.,
          (-5., 5.),
          lambda x: x**2,
          (9., 6., 2.),
          (9., 6., (1., 1.))
      ),
  )
  def test_sharp_quadratic_enclosure_even_symmetric_hessian(
      self, x0, trust_region, sigma, taylor_coefficients_at_x0, expected):
    for np_like in self.backends:
      actual = (
          primitive_enclosures.sharp_quadratic_enclosure_even_symmetric_hessian(
              x0, trust_region, sigma, taylor_coefficients_at_x0, np_like)
      )
      self.assert_enclosure_equal(expected, actual)


if __name__ == '__main__':
  absltest.main()
