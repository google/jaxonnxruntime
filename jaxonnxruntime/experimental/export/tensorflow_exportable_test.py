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

"""Tests for tensorflow_exportable."""

from absl import logging
from absl.testing import absltest
import chex
import jax
from jaxonnxruntime.experimental.export import exportable_test_utils
from jaxonnxruntime.experimental.export import tensorflow_exportable
import tensorflow as tf


class TensorflowExportableObjTest(exportable_test_utils.ExportableTestCase):

  def setUp(self):
    super().setUp()

    def tf_func(*args, **kwargs):
      x, y = args
      a, b = kwargs['a'], kwargs['b']
      res1 = tf.linalg.matmul(a, x)
      res2 = tf.linalg.matmul(b, y)
      res = res1 + res2
      return res

    args = (
        tf.ones((2, 3), dtype=tf.float32),
        tf.ones((2, 3), dtype=tf.float32),
    )
    kwargs = {
        'a': tf.ones((3, 2), dtype=tf.float32),
        'b': tf.ones((3, 2), dtype=tf.float32),
    }
    self.tf_exportable = tensorflow_exportable.TensorflowExportable(
        tf_func, args, kwargs, ['cpu']
    )
    self.args = args
    self.kwargs = kwargs

  def test_name(self):
    name = self.tf_exportable.fun_name
    self.assertEqual(name, 'tf_func')

  def test_platforms(self):
    self.assertEqual(self.tf_exportable.platforms, ('cpu',))

  def test_tf_platform(self):
    self.assertEqual(self.tf_exportable.tf_platform, 'CPU')

  def test_in_avals(self):
    in_avals = self.tf_exportable.in_avals
    logging.info('in_avals: %s', in_avals)
    self.assertLen(in_avals, 4)

  def test_out_avals(self):
    out_avals = self.tf_exportable.out_avals
    logging.info('out_avals: %s', out_avals)
    self.assertLen(out_avals, 1)

  def test_module_kept_var_idx(self):
    self.assertEqual(self.tf_exportable.module_kept_var_idx, (0, 1, 2, 3))

  def test_in_sharding(self):
    in_sharding = self.tf_exportable.in_shardings_hlo
    self.assertLen(in_sharding, 4)
    self.assertFalse(all(in_sharding))

  def test_out_sharding(self):
    out_sharding = self.tf_exportable.out_shardings_hlo
    self.assertLen(out_sharding, len(self.tf_exportable.out_avals))
    self.assertFalse(all(out_sharding))

  def test_nr_devices(self):
    self.assertLen(jax.devices(), self.tf_exportable.nr_devices)

  def test_mlir_module_str(self):
    mlir_module_str = self.tf_exportable.mlir_module_str
    logging.info('mlir_module: %s', mlir_module_str)

  def test_exported(self):
    exported = self.tf_exportable.export()
    logging.info('exported: %s', exported)
    loaded_exported = self._save_and_load_exported(exported)
    self.assertClassAttributeType(exported, loaded_exported)
    args = jax.tree_util.tree_map(lambda x: x.numpy(), self.args)
    kwargs = jax.tree_util.tree_map(lambda x: x.numpy(), self.kwargs)
    result = exported.call(*args, **kwargs)
    result2 = loaded_exported.call(*args, **kwargs)
    chex.assert_trees_all_close(result, result2)


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  jax.config.update('jax_traceback_filtering', 'off')
  absltest.main()
