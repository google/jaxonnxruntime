# Copyright 2024 The Jaxonnxruntime Authors.
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

"""Tests for tf2export."""

import functools
from typing import Any
from absl.testing import absltest
import chex
import jax
from jax import numpy as jnp
from jax.experimental import export as jax_export
from jax.experimental import jax2tf
from jax.experimental import mesh_utils
from jax.experimental import pjit
from jax.sharding import PartitionSpec as P  # pylint: disable=g-importing-member
from jaxonnxruntime.experimental.export import exportable_test_utils
from jaxonnxruntime.experimental.export import exportable_utils
import numpy as np
import tensorflow as tf

global_vars: dict[str, Any] = {}


def setUpModule():
  exportable_test_utils.set_up_module(global_vars)


def tearDownModule():
  exportable_test_utils.tear_down_module(global_vars)


class ExportedTest(exportable_test_utils.ExportableTestCase):

  def _save_and_load_exported(
      self, exported: jax_export.Exported
  ) -> jax_export.Exported:
    model_path = self.create_tempdir().full_path
    exportable_utils.save_exported(exported, model_path)
    loaded_exported = exportable_utils.load_exported(model_path)
    return loaded_exported

  def test_tf_function(self):
    """Test tensorflow function to Exported via jax2tf.call_tf."""

    def tf_func(x):
      return tf.math.reduce_sum(tf.math.sin(x))

    x = jnp.arange(12, dtype=np.float32).reshape((3, 4))

    exported_inputs = (x,)

    with self.subTest('forward'):
      exported = jax_export.export(jax2tf.call_tf(tf_func))(*exported_inputs)
      loaded_exported = self._save_and_load_exported(exported)

      chex.assert_trees_all_equal(
          jax_export.call(exported)(*exported_inputs),
          jax_export.call(loaded_exported)(*exported_inputs),
      )
      chex.assert_trees_all_close(
          jax_export.call(exported)(*exported_inputs), tf_func(*exported_inputs)
      )

    with self.subTest('grad'):
      exported = jax_export.export(jax.grad(jax2tf.call_tf(tf_func)))(
          *exported_inputs
      )
      loaded_exported = self._save_and_load_exported(exported)
      chex.assert_trees_all_equal(
          jax_export.call(exported)(*exported_inputs),
          jax_export.call(loaded_exported)(*exported_inputs),
      )

  @absltest.skip(
      'Exported module jax_func was lowered for 8 devices and is called in a'
      ' context with 1 devices.'
      'See sponge2/29bfc71b-3730-490b-9d5b-9391f2dd7c3b.'
  )
  def test_pjit_function(self):
    """Test jax function to Exported."""
    devices = mesh_utils.create_device_mesh((4, 2))
    mesh = jax.sharding.Mesh(devices, axis_names=('x', 'y'))

    @functools.partial(
        pjit.pjit, in_shardings=jax.sharding.NamedSharding(mesh, P('x'))
    )
    def jax_func(x):
      return jnp.sum(jnp.sin(x))

    x = jnp.arange(32, dtype=np.float32).reshape((8, 4))
    exported_inputs = (x,)

    result = jax_func(*exported_inputs)
    exported = jax_export.export(jax_func)(*exported_inputs)
    loaded_exported = self._save_and_load_exported(exported)

    with mesh:
      result1 = jax_export.call(exported)(*exported_inputs)
      result2 = jax_export.call(loaded_exported)(*exported_inputs)
    chex.assert_trees_all_equal(result, result1)
    chex.assert_trees_all_close(result1, result2)


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main()
