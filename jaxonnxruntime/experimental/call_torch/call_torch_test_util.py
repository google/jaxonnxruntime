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

"""test utilities for call_torch API."""
import jax
from jaxonnxruntime.core import onnx_utils
from jaxonnxruntime.experimental import call_torch
import numpy as np
import torch


class CallTorchTestCase(onnx_utils.JortTestCase):
  """Base class for CallTorch tests including numerical checks and boilerplate."""

  def assert_call_torch_convert_and_compare(self, test_module, torch_inputs):
    """assert the converted jittable jax function and torch module numerical accuracy."""
    if not isinstance(
        test_module,
        (torch.jit.ScriptModule, torch.jit.ScriptFunction, torch.nn.Module),
    ):
      test_module = torch.jit.trace(
          func=test_module, example_inputs=torch_inputs
      )
    torch_outputs = test_module(*torch_inputs)
    torch_outputs = jax.tree_map(
        call_torch.torch_tensor_to_np_array, torch_outputs
    )
    if isinstance(torch_outputs, np.ndarray):
      torch_outputs = (torch_outputs,)
    jax_inputs = jax.tree_map(call_torch.torch_tensor_to_np_array, torch_inputs)
    jax_fn, jax_params = call_torch.call_torch(test_module, torch_inputs)
    jax_fn = jax.jit(jax_fn)
    jax_outputs = jax_fn(jax_params, jax_inputs)
    self.assert_allclose(torch_outputs, jax_outputs)
    return jax_fn, jax_params, jax_inputs, jax_outputs
