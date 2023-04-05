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

import operator

from absl.testing import absltest
from absl.testing import parameterized
from autobound import polynomials
from autobound import test_utils
import numpy as np


class TestCase(parameterized.TestCase, test_utils.TestCase):

  @parameterized.parameters(
      ([], .5, operator.mul, operator.pow, 0),
      ([1], .5, operator.mul, operator.pow, 1),
      ([1, 10], .5, operator.mul, operator.pow, 6),
      ([1, 10, 100], .5, operator.mul, operator.pow, 31),
      (
          (np.array([2, 3]), np.eye(2)),
          np.array([-7, 5]),
          lambda a, b: np.tensordot(a, b, np.ndim(b)),
          lambda a, b: np.tensordot(a, b, 0),
          np.array([-5, 8]),
      ),
  )
  def test_eval_polynomial(self, coefficients, z, inner_product, outer_product,
                           expected):
    actual = polynomials.eval_polynomial(coefficients, z, inner_product,
                                         outer_product)
    np.testing.assert_allclose(actual, expected)

  @parameterized.parameters(
      ((1,), 3, 1),
      ((1, 2), 3, 7),
      ((1, 2, 3), 3, 34),
      ((1, 2, (3, 4)), 3, (34, 43)),
      (
          (
              np.array([5, 7]),
          ),
          np.array([-1, -2]),
          np.array([5, 7]),
      ),
      (
          (
              np.array([5, 7]),
              np.array([3, 4]),
          ),
          np.array([-1, -2]),
          np.array([2, -1]),
      ),
      (
          (
              np.array([5, 7]),
              (
                  np.array([-3, -2]),
                  np.array([3, 4]),
              )
          ),
          np.array([-1, -2]),
          (
              np.array([2, -1]),
              np.array([8, 11]),
          ),
      ),
      (
          (0, 0, (0, 1)),
          (-1., 1.),
          (0., 1.),
      ),
      (
          (0, 0, (-1, 1)),
          (-1., 1.),
          (-1., 1.),
      ),
  )
  def test_eval_elementwise_taylor_enclosure(self, enclosure, x_minus_x0,
                                             expected):
    actual = polynomials.eval_elementwise_taylor_enclosure(
        enclosure, x_minus_x0, np)
    self.assert_interval_equal(expected, actual)

  @parameterized.parameters(
      (
          (np.array([2, 3]), np.eye(2)),
          np.array([-7, 5]),
          np.array([-5., 8.]),
      ),
      (
          (np.array([2, 3]), (np.eye(2), 2*np.eye(2))),
          np.array([-7, 5]),
          (np.array([-12., 8.]), np.array([-5., 13.])),
      ),
      (
          (0, 0, (0, 1)),
          (-1., 1.),
          (0., 1.),
      ),
      (
          (0, 0, (-1, 1)),
          (-1., 1.),
          (-1., 1.),
      ),
  )
  def test_eval_taylor_enclosure(self, enclosure, x_minus_x0, expected):
    actual = polynomials.eval_taylor_enclosure(enclosure, x_minus_x0, np)
    self.assert_interval_equal(expected, actual)

  @parameterized.parameters(
      (
          (2, 3),
          (5, 7),
          operator.add,
          0,
          lambda c0, c1, i, j: c0*c1,
          (10, 29, 21),
      )
  )
  def test_arbitrary_bilinear(
      self, a, b, add, additive_identity, term_product_coefficient, expected):
    actual = polynomials.arbitrary_bilinear(a, b, add, additive_identity,
                                            term_product_coefficient)
    self.assertEqual(expected, actual)

  @parameterized.parameters(
      (
          1,
          2,
          3,
          [(1, 1, 0)]
      ),
      (
          2,
          2,
          3,
          [(0, 2, 0), (1, 0, 1)]
      ),
  )
  def test_iter_partitions(self, n, m, k, expected_iterates):
    actual_iterates = list(polynomials._iter_partitions(n, m, k))
    self.assertListEqual(expected_iterates, actual_iterates)

  @parameterized.parameters(
      ((2, 3, 5), 0, (1,)),
      ((2, 3, 5), 1, (2, 3, 5)),
      ((2, 3, 5), 2, (4, 12, 29, 30, 25)),
  )
  def test_integer_power(self, a, exponent, expected):
    # TODO(mstreeter): test non-default values of the kwargs.
    actual = polynomials.integer_power(a, exponent)
    self.assertEqual(expected, actual)

  @parameterized.parameters(
      ([2, 0], 1),
      ([1, 1], 2),
      ([99, 1], 100),
      ([98, 2], 100*99 // 2),
      ([95, 2, 3], 100*99*98*97*96 // (3*2*2)),
  )
  def test_multinomial_coefficient(self, ks, expected):
    actual = polynomials._multinomial_coefficient(ks)
    self.assertEqual(expected, actual)


if __name__ == '__main__':
  absltest.main()
