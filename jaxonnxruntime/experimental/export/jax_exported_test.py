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
from absl import logging
from absl.testing import absltest
import chex
import jax
from jax import export as jax_export
from jax import numpy as jnp
from jax.experimental import jax2tf
from jax.experimental import mesh_utils
from jax.experimental import pjit
from jax.sharding import PartitionSpec as P  # pylint: disable=g-importing-member
from jaxonnxruntime.experimental.export import exportable_test_utils
import numpy as np
import tensorflow as tf


def setUpModule():
  chex.set_n_cpu_devices(8)


class ExportedTest(exportable_test_utils.ExportableTestCase):

  def check_exported_call(self, exported: jax_export.Exported, *args, **kwargs):
    logging.info('exported.__dict__: %s', exported.__dict__)
    f = exported.call
    f = jax.jit(f)
    lowered = f.lower(*args, **kwargs)
    lowering = lowered._lowering
    compile_args = lowering.compile_args
    mlir_module_str = lowering.as_text()
    logging.info('compile_args: %s', compile_args)
    logging.info('mlir_module_str: %s', mlir_module_str)

  def test_jax2tf_call_tf(self):
    """Test tensorflow function to Exported via jax2tf.call_tf."""

    def tf_func(x):
      return tf.math.reduce_sum(tf.math.sin(x))

    x = jnp.arange(12, dtype=np.float32).reshape((3, 4))

    exported_inputs = (x,)

    with self.subTest('forward'):
      jit_func = jax.jit(jax2tf.call_tf(tf_func))
      exported = jax_export.export(jit_func)(*exported_inputs)
      loaded_exported = self._save_and_load_exported(exported)

      chex.assert_trees_all_equal(
          exported.call(*exported_inputs),
          loaded_exported.call(*exported_inputs),
      )
      chex.assert_trees_all_close(
          exported.call(*exported_inputs), tf_func(*exported_inputs)
      )

    with self.subTest('grad'):
      jit_func = jax.jit(jax.grad(jax2tf.call_tf(tf_func)))
      exported = jax_export.export(jit_func)(*exported_inputs)
      loaded_exported = self._save_and_load_exported(exported)
      chex.assert_trees_all_equal(
          exported.call(*exported_inputs),
          loaded_exported.call(*exported_inputs),
      )

  def test_jax_pjit_func(self):
    """Test jax function to Exported."""
    devices = mesh_utils.create_device_mesh((4, 2))
    mesh = jax.sharding.Mesh(devices, axis_names=('x', 'y'))

    @functools.partial(
        pjit.pjit,
        in_shardings=(
            jax.sharding.NamedSharding(mesh, P('x')),
            jax.sharding.NamedSharding(mesh, P('y')),
        ),
    )
    def jax_func(x, y):
      return jnp.sum(jnp.sin(x) + jnp.cos(y))

    x = jnp.arange(32, dtype=np.float32).reshape((4, 8))
    y = jnp.arange(32, dtype=np.float32).reshape((4, 8))
    exported_inputs = (x, y)
    exported_inputs_sharded = (
        jax.device_put(x, jax.sharding.NamedSharding(mesh, P('x'))),
        jax.device_put(y, jax.sharding.NamedSharding(mesh, P('y'))),
    )

    result = jax_func(*exported_inputs)
    exported = jax_export.export(jax_func)(*exported_inputs)
    loaded_exported = self._save_and_load_exported(exported)
    logging.info('check exported.\n\n')
    # Need use the sharded JAX array inputs.
    self.check_exported_call(exported, *exported_inputs_sharded)
    logging.info('check loaded_expoted.\n\n')
    self.check_exported_call(loaded_exported, *exported_inputs_sharded)

    result1 = exported.call(*exported_inputs_sharded)
    result2 = loaded_exported.call(*exported_inputs_sharded)
    chex.assert_trees_all_equal(result, result1)
    chex.assert_trees_all_equal(result1, result2)


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  jax.config.update('jax_traceback_filtering', 'off')
  absltest.main()
