# Copyright 2025 The Jaxonnxruntime Authors.
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

"""Tests for jax exportable."""

import os
from typing import Any
from absl.testing import absltest
import chex
import jax
from jax import numpy as jnp
from jaxonnxruntime.experimental.export import exportable
from jaxonnxruntime.experimental.export import exportable_test_utils
import numpy as np


class ExportableTest(exportable_test_utils.ExportableTestCase):

  def test_basic(self):
    def jax_func(x):
      return jnp.sum(jnp.sin(x))

    x = jnp.arange(32, dtype=np.float32).reshape((8, 4))
    exported_inputs = (x,)

    exportable_obj = exportable.Exportable(
        jax_func, exported_inputs, {}, ['cpu', 'cuda', 'rocm', 'tpu']
    )
    exported = exportable_obj.export()
    loaded_exported = self._save_and_load_exported(exported)
    self.assertClassAttributeType(exported, loaded_exported)

    result = exported.call(*exported_inputs)
    result2 = loaded_exported.call(*exported_inputs)
    chex.assert_trees_all_close(result, result2)


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  os.environ['XLA_FLAGS'] = (
      '--xla_force_host_platform_device_count=8'  # Use 8 CPU devices
  )
  absltest.main()
