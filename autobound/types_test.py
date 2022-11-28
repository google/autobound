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

from absl.testing import absltest
from autobound import test_utils
from autobound import types
import jax.numpy as jnp
import numpy as np
import tensorflow.experimental.numpy as tnp


class TestCase(test_utils.TestCase):

  def test_ndarray(self):
    self.assertIsInstance(np.eye(3), types.NDArray)
    self.assertIsInstance(jnp.eye(3), types.NDArray)
    self.assertIsInstance(tnp.eye(3), types.NDArray)

  def test_numpy_like(self):
    self.assertIsInstance(np, types.NumpyLike)
    self.assertIsInstance(jnp, types.NumpyLike)
    self.assertIsInstance(tnp, types.NumpyLike)


if __name__ == '__main__':
  absltest.main()
