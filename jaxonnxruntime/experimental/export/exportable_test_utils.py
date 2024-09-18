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

import inspect
from typing import Any
from absl import logging
from absl.testing import parameterized
import jax
from jax import export as jax_export
from jaxonnxruntime.experimental.export import exportable_utils


class ExportableTestCase(parameterized.TestCase):
  """Base class for Exportable tests."""

  def _save_and_load_exported(
      self, exported: jax_export.Exported
  ) -> jax_export.Exported:
    model_path = self.create_tempdir().full_path
    exportable_utils.save_exported(exported, model_path)
    loaded_exported = exportable_utils.load_exported(model_path)
    return loaded_exported

  def check_exported_call(self, exported: jax_export.Exported, *args, **kwargs):
    logging.info('exported.__dict__: %s', exported.__dict__)
    f = exported.call
    f = jax.jit(f)
    lowered = f.lower(*args, **kwargs)
    lowering = lowered._lowering  # pylint: disable=protected-access
    compile_args = lowering.compile_args
    mlir_module_str = lowering.as_text()
    logging.info('compile_args: %s', compile_args)
    logging.info('mlir_module_str: %s', mlir_module_str)

  def assertClassAttributeType(self, obj: Any, other_obj: Any):  # pylint: disable=invalid-name
    def get_attributes_and_types_inspect(obj):
      type_dict = {}
      for name, value in inspect.getmembers(obj):
        if not name.startswith('__'):
          type_dict[name] = type(value)
      return type_dict

    obj_type_dict = get_attributes_and_types_inspect(obj)
    other_obj_type_dict = get_attributes_and_types_inspect(other_obj)
    for k in obj_type_dict:
      self.assertEqual(
          obj_type_dict[k],
          other_obj_type_dict[k],
          f'Exported {k} does not match loaded exported'
          f' {other_obj_type_dict[k]}',
      )
