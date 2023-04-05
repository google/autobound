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

from absl.testing import absltest
from absl.testing import parameterized
from autobound import interval_arithmetic
from autobound import test_utils
import numpy as np


class TestCase(parameterized.TestCase, test_utils.TestCase):

  @parameterized.parameters(
      ((1, 2), (10, 20), (11, 22)),
      (1, (10, 20), (11, 21)),
      ((1, 2), 10, (11, 12)),
      (1, 10, 11),
      (10, np.array([1, 2]), np.array([11, 12])),
      (np.array([1, 2]), 10, np.array([11, 12])),
      (
          (np.array([1, 2]), np.array([3, 4])),
          np.array([10, 20]),
          (np.array([11, 22]), np.array([13, 24]))
      ),
      (
          np.array([10, 20]),
          (np.array([1, 2]), np.array([3, 4])),
          (np.array([11, 22]), np.array([13, 24]))
      ),
      (
          (np.array([1, 2]), np.array([3, 4])),
          (np.array([10, 20]), np.array([30, 40])),
          (np.array([11, 22]), np.array([33, 44]))
      ),
  )
  def test_add(self, a, b, expected):
    for np_like in self.backends:
      actual = interval_arithmetic.IntervalArithmetic(np_like).add(a, b)
      self.assert_interval_equal(expected, actual)

  @parameterized.parameters(
      # Test multiplication of scalar intervals via arbitary_bilinear(), using
      # all 9 combinations of signs for the interval end points.
      #
      # Test with both assume_product=True and assume_product=False.  This only
      # changes the expected output in 1 of the 9 cases.
      ((2, 3), (5, 7), lambda np_like: np_like.multiply, False, (10, 21)),
      ((2, 3), (5, 7), lambda np_like: np_like.multiply, True, (10, 21)),
      ((-3, -2), (5, 7), lambda np_like: np_like.multiply, False, (-21, -10)),
      ((-3, -2), (5, 7), lambda np_like: np_like.multiply, True, (-21, -10)),
      ((-2, 3), (5, 7), lambda np_like: np_like.multiply, False, (-14, 21)),
      ((-2, 3), (5, 7), lambda np_like: np_like.multiply, True, (-14, 21)),
      ((2, 3), (-7, -5), lambda np_like: np_like.multiply, False, (-21, -10)),
      ((2, 3), (-7, -5), lambda np_like: np_like.multiply, True, (-21, -10)),
      ((-3, -2), (-7, -5), lambda np_like: np_like.multiply, False, (10, 21)),
      ((-3, -2), (-7, -5), lambda np_like: np_like.multiply, True, (10, 21)),
      ((-2, 3), (-7, -5), lambda np_like: np_like.multiply, False, (-21, 14)),
      ((-2, 3), (-7, -5), lambda np_like: np_like.multiply, True, (-21, 14)),
      ((2, 3), (-5, 7), lambda np_like: np_like.multiply, False, (-15, 21)),
      ((2, 3), (-5, 7), lambda np_like: np_like.multiply, True, (-15, 21)),
      ((-3, -2), (-5, 7), lambda np_like: np_like.multiply, False, (-21, 15)),
      ((-3, -2), (-5, 7), lambda np_like: np_like.multiply, True, (-21, 15)),
      # This is the one case where setting assume_product=False yields a looser
      # interval.
      ((-2, 3), (-5, 7), lambda np_like: np_like.multiply, False, (-29, 31)),
      ((-2, 3), (-5, 7), lambda np_like: np_like.multiply, True, (-15, 21)),
      # Test other bilinear operations.
      (
          (np.array([2, 3]), np.array([5, 7])),
          (np.array([11, 13]), np.array([17, 19])),
          lambda np_like: np_like.dot,
          False,
          (61, 218),
      ),
      (
          (np.diag([2, 3]), np.diag([5, 7])),
          (np.array([11, 13]), np.array([17, 19])),
          lambda np_like: np_like.matmul,
          False,
          (np.array([22, 39]), np.array([85, 133])),
      ),
      (
          np.ones((1, 3)),
          (np.zeros((3,)), np.ones((3,))),
          lambda np_like: functools.partial(np_like.tensordot, axes=1),
          False,
          (np.array([0.]), np.array([3.]))
      ),
      (
          (np.zeros((1, 3)), np.ones((1, 3))),
          np.ones((3,)),
          lambda np_like: functools.partial(np_like.tensordot, axes=1),
          False,
          (np.array([0.]), np.array([3.]))
      ),
      (
          (np.zeros((1, 3)), np.ones((1, 3))),
          (np.zeros((3,)), np.ones((3,))),
          lambda np_like: functools.partial(np_like.tensordot, axes=1),
          False,
          (np.array([0.]), np.array([3.]))
      ),
  )
  def test_arbitrary_bilinear(self, a, b, get_bilinear, assume_product,
                              expected):
    for np_like in self.backends:
      bilinear = get_bilinear(np_like)
      arithmetic = interval_arithmetic.IntervalArithmetic(np_like)
      actual = arithmetic.arbitrary_bilinear(a, b, bilinear, assume_product)
      self.assert_interval_equal(expected, actual)

  @parameterized.parameters(
      ([1, 2, 3], 1, [1, 2, 3]),
      ([1, 2, 3], 2, [[1, 0, 0], [0, 2, 0], [0, 0, 3]]),
  )
  def test_generalized_diag_interval(self, a, n, expected):
    for np_like in self.backends:
      arithmetic = interval_arithmetic.IntervalArithmetic(np_like)
      actual = arithmetic._generalized_diag_interval(a, n)
      self.assert_interval_equal(expected, actual)

  @parameterized.parameters(
      (2, 5, 10),
      ((2, 3), 5, (10, 15)),
      (2, (5, 7), (10, 14)),
      (np.array([2, 3]), np.array([5, 7]), np.array([10, 21])),
      (np.array([2, 3]), (5, 7), (np.array([10, 15]), np.array([14, 21]))),
      ((5, 7), np.array([2, 3]), (np.array([10, 15]), np.array([14, 21]))),
      # Test all 9 valid combinations of signs for scalar intervals.
      ((2, 3), (5, 7), (10, 21)),
      ((-3, -2), (5, 7), (-21, -10)),
      ((-2, 3), (5, 7), (-14, 21)),
      ((2, 3), (-7, -5), (-21, -10)),
      ((-3, -2), (-7, -5), (10, 21)),
      ((-2, 3), (-7, -5), (-21, 14)),
      ((2, 3), (-5, 7), (-15, 21)),
      ((-3, -2), (-5, 7), (-21, 15)),
      ((-2, 3), (-5, 7), (-15, 21)),
      # Same 9 combinations, but as a single test.
      (
          (
              np.array([2, -3, -2, 2, -3, -2, 2, -3, -2]),
              np.array([3, -2, 3, 3, -2, 3, 3, -2, 3])
          ),
          (
              np.array([5, 5, 5, -7, -7, -7, -5, -5, -5]),
              np.array([7, 7, 7, -5, -5, -5, 7, 7, 7])
          ),
          (
              np.array([10, -21, -14, -21, 10, -21, -15, -21, -15]),
              np.array([21, -10, 21, -10, 21, 14, 21, 15, 21])
          ),
      ),
  )
  def test_multiply(self, a, b, expected):
    for np_like in self.backends:
      actual = interval_arithmetic.IntervalArithmetic(np_like).multiply(a, b)
      self.assert_interval_equal(expected, actual)

  @parameterized.parameters(
      ((1, 2), (-2, -1)),
      (np.array([-1, 2]), np.array([1, -2])),
      (
          (np.array([-1, 2]), np.array([10, 20])),
          (np.array([-10, -20]), np.array([1, -2])),
      ),
  )
  def test_negative(self, a, expected):
    for np_like in self.backends:
      actual = interval_arithmetic.IntervalArithmetic(np_like).negative(a)
      self.assert_interval_equal(expected, actual)

  @parameterized.parameters(
      ((-1., 1.), 2, 0, (0., 1.)),
      (
          (np.array([-1, -2]), np.array([3, 4])),
          2,
          0,
          (np.array([[0, -6], [-6, 0]]), np.array([[9, 12], [12, 16]]))
      ),
      ([2., 3.], 2, 1, np.array([4., 9.])),
  )
  def test_outer_power(self, a, exponent, batch_dims, expected):
    for np_like in self.backends:
      actual = interval_arithmetic.IntervalArithmetic(np_like).outer_power(
          a, exponent, batch_dims)
      self.assert_interval_equal(expected, actual)

  @parameterized.parameters(
      # ndarray times ndarray
      (np.array([2, 3]), np.array([5, 7]), 0, np.array([[10, 14], [15, 21]])),
      (np.array([2, 3]), np.array([5, 7]), 1, np.array([10, 21])),
      # ndarray times interval
      (
          np.array([1, 2]),
          (np.array([3, 4]), np.array([5, 6])),
          0,
          (np.array([[3, 4], [6, 8]]), np.array([[5, 6], [10, 12]])),
      ),
      # interval times ndarray
      (
          (np.array([3, 4]), np.array([5, 6])),
          np.array([1, 2]),
          0,
          (np.array([[3, 6], [4, 8]]), np.array([[5, 10], [6, 12]])),
      ),
      # interval times interval
      (
          (np.array([-1, 2]), np.array([1, 2])),
          (np.array([3, 4]), np.array([5, 6])),
          0,
          (np.array([[-5, -6], [6, 8]]), np.array([[5, 6], [10, 12]])),
      ),
  )
  def test_outer_product(self, a, b, batch_dims, expected):
    for np_like in self.backends:
      actual = interval_arithmetic.IntervalArithmetic(np_like).outer_product(
          a, b, batch_dims)
      self.assert_interval_equal(expected, actual)

  @parameterized.parameters(
      ((-2., 3.), 2, (0., 9.)),
      ((2., 3.), 2, (4., 9.)),
      ((-3., -2.), 2, (4., 9.)),
      ((-2., 3.), 3, (-8., 27.)),
      ((2., 3.), 3, (8., 27.)),
      ((-3., -2.), 3, (-27., -8.)),
      ((4., 9.), .5, (2., 3.)),
      (
          (np.array([-2., 2., -3.]), np.array([3., 3., -2.])),
          2,
          (np.array([0., 4., 4.]), np.array([9., 9., 9.]))
      ),
  )
  def test_power(self, a, exponent, expected):
    for np_like in self.backends:
      actual = interval_arithmetic.IntervalArithmetic(np_like).power(a,
                                                                     exponent)
      self.assert_interval_equal(expected, actual)

  @parameterized.parameters(
      ((10, 20), (1, 2), (8, 19)),
      (10, (1, 2), (8, 9)),
      ((10, 20), 1, (9, 19)),
      (10, 1, 9),
      (10, np.array([1, 2]), np.array([9, 8])),
      (np.array([10, 20]), 1, np.array([9, 19])),
      (
          (np.array([10, 20]), np.array([30, 40])),
          np.array([1, 2]),
          (np.array([9, 18]), np.array([29, 38]))
      ),
      (
          np.array([10, 20]),
          (np.array([1, 2]), np.array([3, 4])),
          (np.array([7, 16]), np.array([9, 18]))
      ),
      (
          (np.array([10, 20]), np.array([30, 40])),
          (np.array([1, 2]), np.array([3, 4])),
          (np.array([7, 16]), np.array([29, 38]))
      ),
  )
  def test_subtract(self, a, b, expected):
    for np_like in self.backends:
      actual = interval_arithmetic.IntervalArithmetic(np_like).subtract(a, b)
      self.assert_interval_equal(expected, actual)

  @parameterized.parameters(
      (2, 3, 0, 6),
      (np.array([2, 3]), np.array([5, 7]), 0, np.array([[10, 14], [15, 21]])),
      (np.array([2, 3]), np.array([5, 7]), 1, 31),
      ((-.5, .5), (-.5, .5), 0, (-.25, .25)),
  )
  def test_tensordot(self, a, b, axes, expected):
    for np_like in self.backends:
      actual = interval_arithmetic.IntervalArithmetic(np_like).tensordot(
          a, b, axes)
      self.assert_interval_equal(expected, actual)


if __name__ == '__main__':
  absltest.main()
