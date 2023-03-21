# Copyright 2022 The autobound Authors.
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

"""Base class for unit tests."""

import math
from typing import List

from absl.testing import absltest
from autobound import types
import numpy as np


MAX_SIGMOID_DERIV = .25
MIN_SIGMOID_SECOND_DERIV = -0.09622504486493762
MAX_SIGMOID_SECOND_DERIV = 0.09622504486493762
MIN_SIGMOID_THIRD_DERIV = -0.125
MAX_SIGMOID_THIRD_DERIV = 0.04166666666666668


def sigmoid(x):
  return 1/(1+math.exp(-x)) if x >= 0 else math.exp(x)/(1+math.exp(x))


def sigmoid_deriv(x):
  return sigmoid(x)*sigmoid(-x)


def sigmoid_second_deriv(x):
  s = sigmoid(x)
  return s*sigmoid(-x)*(1-2*s)


def sigmoid_third_deriv(x):
  s = sigmoid(x)
  sm = sigmoid(-x)
  return s*sm*((1-2*s)**2 - 2*s*sm)

def softplus(x):
  # Avoid overflow for large positive x using:
  # log(1+exp(x)) == log(1+exp(-|x|)) + max(x, 0).
  return math.log1p(math.exp(-abs(x))) + max(x, 0.)


def swish(x):
  return x*sigmoid(x)


class TestCase(absltest.TestCase):
  """Base class for test cases."""

  def assert_allclose_strict(self, expected, actual, **kwargs):
    """Like np.testing.assert_allclose, but requires same shape/dtype."""
    np.testing.assert_allclose(actual, expected, **kwargs)
    self.assertEqual(np.asarray(expected).shape, np.asarray(actual).shape,
                     (expected, actual))
    self.assertEqual(np.asarray(expected).dtype, np.array(actual).dtype,
                     (expected, actual))

  def assert_enclosure_equal(self, expected, actual, **kwargs):
    self.assertLen(actual, len(expected))
    for e, a in zip(expected, actual):
      self.assert_interval_equal(e, a, **kwargs)

  def assert_interval_equal(self, expected, actual, **kwargs):
    if isinstance(expected, tuple):
      e0, e1 = expected
      self.assertIsInstance(actual, tuple)
      self.assertLen(actual, 2)
      a0, a1 = actual
      self.assert_allclose_strict(e0, a0, **kwargs)
      self.assert_allclose_strict(e1, a1, **kwargs)
    else:
      self.assert_allclose_strict(expected, actual, **kwargs)

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.backends = _get_backends()


def _get_backends() -> List[types.NumpyLike]:
  """Returns list of NumpyLike back ends to test."""
  backends = [np]

  try:
    from jax.config import config as jax_config
    import jax.numpy as jnp
    backends.append(jnp)
    jax_config.update('jax_enable_x64', True)
  except ModuleNotFoundError:
    pass

  try:
    import tensorflow.experimental.numpy as tnp
    tnp.experimental_enable_numpy_behavior()
    backends.append(tnp)
  except ModuleNotFoundError:
    pass

  return backends