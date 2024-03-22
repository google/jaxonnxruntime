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
import os
from typing import Any
from absl import logging
from absl.testing import parameterized
import jax
from jax.experimental import export as jax_export
from jax.lib import xla_bridge
from jaxonnxruntime.experimental.export import exportable_utils


def set_up_module(global_vars: dict[str, Any]):
  """Set up module for exportable tests."""
  global_vars['prev_xla_flags'] = os.getenv('XLA_FLAGS')
  flags_str = global_vars['prev_xla_flags'] or ''
  # Don't override user-specified device count, or other XLA flags.
  if 'xla_force_host_platform_device_count' not in flags_str:
    os.environ['XLA_FLAGS'] = (
        flags_str + ' --xla_force_host_platform_device_count=8'
    )
  # Clear any cached backends so new CPU backend will pick up the env var.
  xla_bridge.get_backend.cache_clear()
  global_vars['prev_spmd_lowering_flag'] = (
      jax.config.experimental_xmap_spmd_lowering
  )
  jax.config.update('experimental_xmap_spmd_lowering', True)


def tear_down_module(global_vars: dict[str, Any]):
  """Tear down module for exportable tests."""
  if global_vars.get('prev_xla_flags') is None:
    del os.environ['XLA_FLAGS']
  else:
    os.environ['XLA_FLAGS'] = global_vars['prev_xla_flags']
  xla_bridge.get_backend.cache_clear()
  jax.config.update(
      'experimental_xmap_spmd_lowering', global_vars['prev_spmd_lowering_flag']
  )


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
    f = jax_export.call(exported)
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
