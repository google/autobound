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
from autobound import enclosure_arithmetic
from autobound import test_utils
import numpy as np


class TestCase(parameterized.TestCase, test_utils.TestCase):

  @parameterized.parameters(
      (100, (0, .25), (1,), (10,), (11,)),
      (100, (0, .25), (1, 2), (10,), (11, 2)),
      (100, (0, .25), (1, 2), (10, 20), (11, 22)),
      (100, (0, .25), (1,), (10, 20), (11, 20)),
      (100, (0, .25), (1, (2, 3),), (10, (20, 30),), (11, (22, 33),)),
      # Test a case where we truncate the sum.
      # 1+10+(2+20)*.25 == 16.5
      (0, (0, .25), (1, 2), (10., 20.), ((11., 16.5),)),
      (
          100,
          (np.array([0, 0,]), np.array([1, 1])),
          (np.array([1, 2]), 3*np.eye(2)),
          (np.array([10, 20]), 30*np.eye(2)),
          (np.array([11, 22]), 33*np.eye(2)),
      ),
  )
  def test_add(self, max_degree, trust_region, a, b, expected):
    for np_like in self.backends:
      arithmetic = enclosure_arithmetic.TaylorEnclosureArithmetic(
          max_degree, trust_region, np_like)
      actual = arithmetic.add(a, b)
      self.assert_enclosure_equal(expected, actual)

  @parameterized.parameters(
      (
          100,
          (0, .5),
          np,
          ([2],),
          ([3],),
          lambda u, v, p, q: np.einsum('i...,i...->i...', u, v),
          ([6],)
      ),
  )
  def test_arbitrary_bilinear(self, max_degree, trust_region, np_like, a, b,
                              pairwise_batched_bilinear, expected):
    arithmetic = enclosure_arithmetic.TaylorEnclosureArithmetic(
        max_degree, trust_region, np_like)
    actual = arithmetic.arbitrary_bilinear(a, b, pairwise_batched_bilinear)
    self.assert_enclosure_equal(expected, actual)

  @parameterized.parameters(
      (
          2,
          (0, .5),
          np,
          (2, 3),
          (5, 7),
          # 2+3(7z) = 2+21z
          (2, 21)
      ),
      (
          2,
          (0, .5),
          np,
          (3, 3, (2, 4)),
          (1, 1),
          # a(x) = 1 + 1(x-x0).
          # a(x)-a(x0) = x
          (3, 3, (2, 4)),
      ),
      (
          2,
          (-10, 10),
          np,
          (1, (-1, 1)),
          (-1, 1),
          (1, (-1, 1)),
      ),
      (
          2,
          (np.array([-10, -10, -10]), np.array([10, 10, 10])),
          np,
          (np.array([1, 0, 2]), (np.array([-1, -1, -1]), np.array([1, 1, 1]))),
          (np.array([-1, 0, 2]), np.eye(3)),
          (np.array([1, 0, 2]), (-np.eye(3), np.eye(3))),
      ),
      (
          2,
          (np.zeros((1,)), np.ones((1,))),
          np,
          (np.zeros((2, 3)), np.zeros((2, 3))),
          (np.zeros((2, 3)), np.zeros((2, 3, 1))),
          (np.zeros((2, 3)), np.zeros((2, 3, 1))),
      ),
  )
  def test_compose_enclosures(
      self, max_degree, trust_region, np_like, scalar_enclosure, arg_enclosure,
      expected):
    arithmetic = enclosure_arithmetic.TaylorEnclosureArithmetic(
        max_degree, trust_region, np_like)
    actual = arithmetic.compose_enclosures(scalar_enclosure, arg_enclosure)
    self.assert_enclosure_equal(expected, actual)

  @parameterized.parameters(
      ((2,), (0, .5), 0, (2,)),
      ((2, 3), (0, .5), 1, (2, 3)),
      ((2, 3), (0, .5), 0, ((2., 3.5),)),
      ((11, 22), (0, .25), 0, ((11., 16.5),)),  # 11+22/4 == 16.5
      # A multivariate linear enclosure, truncated to rank 0.
      (
          (np.array([2., 3.]), np.diag([20, 30])),
          (np.array([0, 0]), np.array([.25, .5])),
          0,
          ((np.array([2., 3.]), np.array([7., 18.])),)
      ),
  )
  def test_enclose_enclosure(self, enclosure, trust_region, max_degree,
                             expected):
    for np_like in self.backends:
      actual = enclosure_arithmetic.enclose_enclosure(
          enclosure, trust_region, max_degree, np_like)
      self.assert_enclosure_equal(expected, actual)

  @parameterized.parameters(
      (np.array(0.), 3, None, np.zeros((1, 1, 1))),
      (np.zeros((3, 5)), 2, None, np.zeros((3, 5, 1, 1))),
      (np.zeros((3, 5)), 2, 1, np.zeros((3, 1, 1, 5))),
  )
  def test_expand_multiple_dims(self, a, n, axis, expected):
    actual = enclosure_arithmetic.expand_multiple_dims(a, n, axis)
    self.assert_allclose_strict(expected, actual)

  @parameterized.parameters(
      (100, (0, .5), (2,), (3,), (6,)),
      (0, (0, .5), (2,), (3,), (6,)),
      (100, (0, .5), (2, 3), (5,), (10, 15)),
      (0, (0, .5), (2, 3), (5,), ((10., 17.5),)),
      (0, (0, .25), (2,), (np.ones((3,)),), (2*np.ones((3,)),)),
      (2, (-10, 10), (0,), ((-1, 1),), ((0, 0),)),
      # Multiplication of degree-0 enclosures should work like np.multiply,
      # and should support broadcasting.
      (0, (0, .5), (np.ones((2, 3)),), (5,), (5*np.ones((2, 3)),)),
      (0, (0, .5), (np.ones((2, 3)),), (np.array([20., 30., 50.]),),
       (np.array([[20., 30., 50.], [20., 30., 50.]]),)),
      (100, (0, .5), (2, 3), (0, 1), (0, 2, 3)),
      (100, (0, .5), (2, [3]), (0, 1), (0, np.array([2]), np.array([3]))),
      (
          100,
          (np.zeros((2,)), np.ones((2,))),
          (np.zeros((2,)), np.eye(2)),
          (np.zeros((2,)), np.eye(2)),
          (np.zeros((2,)), np.zeros((2, 2)),
           np.array([[[1., 0.], [0., 0.]], [[0., 0.], [0., 1.]]]))
      ),
      (100, (-13., 17.), ((-.5, .5),), ((-.5, .5),), ((-.25, .25),)),
  )
  def test_multiply(self, max_degree, trust_region, a, b, expected):
    for np_like in self.backends:
      arithmetic = enclosure_arithmetic.TaylorEnclosureArithmetic(
          max_degree, trust_region, np_like)
      actual = arithmetic.multiply(a, b)
      self.assert_enclosure_equal(expected, actual)

  @parameterized.parameters(
      (100, (np.zeros((2,)), np.ones((2,))), (np.array([-1, 2]),),
       (np.array([1, -2]),)),
      (100, (0, .25), (1, 2), (-1, -2)),
      (0, (0, .25), (1, 2), ((-1.5, -1.),)),
  )
  def test_negative(self, max_degree, trust_region, a, expected):
    for np_like in self.backends:
      arithmetic = enclosure_arithmetic.TaylorEnclosureArithmetic(
          max_degree, trust_region, np_like)
      actual = arithmetic.negative(a)
      self.assert_enclosure_equal(expected, actual)

  @parameterized.parameters(
      (2, 3, 0, 0, 6),
      (np.zeros((2, 2)), np.zeros((2, 2)), 1, 1, np.zeros((2, 2, 2))),
      (np.zeros((2, 3, 5)), np.zeros((1, 1, 7, 11)), 1, 2,
       np.zeros((2, 3, 5, 7, 11))),
  )
  def test_pairwise_batched_multiply(self, u, v, p, q, expected):
    for np_like in self.backends:
      actual = enclosure_arithmetic._pairwise_batched_multiply(u, v, p, q,
                                                               np_like)
      self.assert_allclose_strict(expected, actual)

  @parameterized.parameters(
      (100, (0, 1), (2,), 0, (1,)),
      (100, (0, 1), (2,), 3, (8,)),
      (
          2,
          (np.zeros((3,)), np.array([1, 2, 3])),
          (np.zeros((3,)), np.eye(3)),
          4,
          (np.zeros((3,)), np.zeros((3, 3)),
           (
               np.zeros((3, 3, 3)),
               np.array([[[1., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                         [[0., 0., 0.], [0., 4., 0.], [0., 0., 0.]],
                         [[0., 0., 0.], [0., 0., 0.], [0., 0., 9.]]])
           )
          )
      ),
      (
          100,
          (np.zeros((3,)), np.ones((3,))),
          (.5*np.ones((3,)),),
          0,
          (np.ones((3,)),)
      ),
  )
  def test_power(self, max_degree, trust_region, enclosure, p, expected):
    for np_like in self.backends:
      arithmetic = enclosure_arithmetic.TaylorEnclosureArithmetic(
          max_degree, trust_region, np_like)
      actual = arithmetic.power(enclosure, p)
      self.assert_enclosure_equal(expected, actual)

  @parameterized.parameters(
      (100, (0, .5), (1, 2, 3), (10, 20), (-9, -18, 3)),
      (1, (0, .5), (1, 2, 3), (10, 20), (-9, (-18., -16.5))),
  )
  def test_subtract(self, max_degree, trust_region, a, b, expected):
    for np_like in self.backends:
      arithmetic = enclosure_arithmetic.TaylorEnclosureArithmetic(
          max_degree, trust_region, np_like)
      actual = arithmetic.subtract(a, b)
      self.assert_enclosure_equal(expected, actual)

  @parameterized.named_parameters(
      (
          'scalar_ndarray',
          2.,
          3.,
          5,
          7,
          0,
          6.
      ),
      (
          'scalar_interval',
          (-2., 3.),
          (-5., 7.),
          11,
          13,
          0,
          (-15., 21.)
      ),
      (
          'vector_ndarray',
          # [2, 3] * <[[5, 0, 0], [0, 0, 7]], z>
          # == <[[10, 0, 0], [0, 0, 21]],  z >
          [2., 3.],
          [[5., 0., 0.], [0., 0., 7.]],
          0,
          1,
          1,
          np.array([[10., 0., 0.], [0., 0., 21.]]),
      )
  )
  def test_elementwise_term_product_coefficient(
      self, c0, c1, i, j, x_ndim, expected):
    for np_like in self.backends:
      actual = enclosure_arithmetic._elementwise_term_product_coefficient(
          c0, c1, i, j, x_ndim, np_like)
      self.assert_interval_equal(expected, actual)

  @parameterized.named_parameters(
      (
          'scalar_ndarray',
          2.,
          3,
          5,
          0,
          32.
      ),
      (
          'scalar_interval',
          (-.5, .5),
          3,
          2,
          0,
          (0., .25)
      ),
      (
          'vector_interval',
          ([-.5, 0.], [.5, 0.]),
          1,
          2,
          1,
          (np.array([[0., 0.], [0., 0.]]), np.array([[0.25, 0.], [0., 0.]]))
      ),
      (
          'vector_ndarray_constant_term',
          [2., 3.],
          0,
          2,
          5,
          np.array([4., 9.]),
      ),
  )
  def test_elementwise_term_power_coefficient(
      self, c, i, exponent, x_ndim, expected):
    for np_like in self.backends:
      actual = enclosure_arithmetic._elementwise_term_power_coefficient(
          c, i, exponent, x_ndim, np_like)
      self.assert_interval_equal(expected, actual)

  @parameterized.parameters(
      (2, 3, 6),
      (
          5 * np.ones((2,)),
          7 * np.ones((2, 3)),
          35 * np.ones((2, 3))
      ),
  )
  def test_left_broadcasting_multiply(self, a, b, expected):
    for np_like in self.backends:
      actual = enclosure_arithmetic._left_broadcasting_multiply(a, b, np_like)
      self.assert_allclose_strict(expected, actual)


if __name__ == '__main__':
  absltest.main()
