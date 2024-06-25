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

"""Tests for torch_exportable."""

from absl import logging
from absl.testing import absltest
import chex
import jax
from jaxonnxruntime.experimental.export import exportable_test_utils
from jaxonnxruntime.experimental.export import exportable_utils
from jaxonnxruntime.experimental.export import torch_exportable
import torch


class TorchExportableObjTest(exportable_test_utils.ExportableTestCase):

  def setUp(self):
    super().setUp()

    args = (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0]))
    kwargs = {}

    def f(x, y):
      return x + y

    torch_module = f

    self.exportable = torch_exportable.TorchExportable(
        torch_module, args, kwargs, ['cpu']
    )
    self.args = args
    self.kwargs = kwargs
    self.torch_module = torch_module

  def test_exportable(self):
    exported = self.exportable.export()
    logging.info('exported: %s', exported)
    loaded_exported = self._save_and_load_exported(exported)
    self.assertClassAttributeType(exported, loaded_exported)
    args = jax.tree_util.tree_map(
        exportable_utils.torch_tensor_to_jax_array, self.args
    )
    kwargs = jax.tree_util.tree_map(
        exportable_utils.torch_tensor_to_jax_array, self.kwargs
    )
    result = exported.call(*args, **kwargs)
    result2 = loaded_exported.call(*args, **kwargs)
    chex.assert_trees_all_close(result, result2)


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  jax.config.update('jax_traceback_filtering', 'off')
  absltest.main()
