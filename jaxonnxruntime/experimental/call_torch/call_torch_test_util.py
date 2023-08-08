# Copyright 2023 The Jaxonnxruntime Authors.
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
import jax
from jaxonnxruntime.experimental import call_torch
import numpy as np
import torch


def is_sequence(x):
  try:
    iter(x)
  except TypeError:
    return False
  else:
    return True


class CallTorchTestCase(absltest.TestCase):
  """Base class for CallTorch tests including numerical checks and boilerplate."""

  def assert_allclose(self, x, y, *, atol=10e-7, rtol=10e-5, err_msg=''):
    """Assert that x and y, either arrays or nested tuples/lists, are close."""
    if isinstance(x, dict):
      self.assertIsInstance(y, dict)
      self.assertEqual(set(x.keys()), set(y.keys()))
      for k in x.keys():
        self.assert_allclose(x[k], y[k], atol=atol, rtol=rtol, err_msg=err_msg)
    elif is_sequence(x) and not hasattr(x, '__array__'):
      self.assertTrue(is_sequence(y) and not hasattr(y, '__array__'))
      self.assertEqual(len(x), len(y))
      for x_elt, y_elt in zip(x, y):
        self.assert_allclose(
            x_elt, y_elt, atol=atol, rtol=rtol, err_msg=err_msg
        )
    elif hasattr(x, '__array__') or np.isscalar(x):
      self.assertTrue(hasattr(y, '__array__') or np.isscalar(y), type(y))
      x = np.asarray(x)
      y = np.asarray(y)
      np.testing.assert_allclose(x, y, atol=atol, rtol=rtol, err_msg=err_msg)
    elif x == y:
      return
    else:
      raise TypeError((type(x), type(y)))

  def assert_call_torch_convert_and_compare(self, test_module, torch_inputs):
    """assert the converted jax function and torch module numerical accuracy."""
    if not isinstance(
        test_module, (torch.jit.ScriptModule, torch.jit.ScriptFunction)
    ):
      test_module = torch.jit.script(test_module, torch_inputs)
    torch_outputs = test_module(*torch_inputs)
    torch_outputs = jax.tree_map(
        call_torch.torch_tensor_to_np_array, torch_outputs
    )
    if isinstance(torch_outputs, np.ndarray):
      torch_outputs = (torch_outputs,)
    jax_inputs = jax.tree_map(call_torch.torch_tensor_to_np_array, torch_inputs)
    jax_fn, jax_params = call_torch.call_torch(test_module, torch_inputs)
    jax_outputs = jax_fn(jax_params, jax_inputs)
    self.assert_allclose(torch_outputs, jax_outputs)
