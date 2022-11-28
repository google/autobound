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

from absl.testing import absltest
from jax.config import config as jax_config
import jax.numpy as jnp
import numpy as np
import tensorflow.experimental.numpy as tnp

# List of NumpyLike back ends used in various unit tests.
BACKENDS = (np, jnp, tnp)


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
    jax_config.update('jax_enable_x64', True)
    tnp.experimental_enable_numpy_behavior()
