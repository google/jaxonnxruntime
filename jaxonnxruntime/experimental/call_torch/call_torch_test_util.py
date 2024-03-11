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

"""test utilities for call_torch API."""

import os
from typing import Any, Optional

from absl import logging
import chex
import jax
from jax import numpy as jnp
from jaxonnxruntime.core import onnx_utils
from jaxonnxruntime.experimental import call_torch
import numpy as np
import torch

import onnx


def _maybe_upcast(x: jax.Array):
  """Upcast bfloat16 array to float32 array if need."""
  if x.dtype in [jnp.bfloat16]:
    return x.astype(np.float32)
  return x


class CallTorchTestCase(onnx_utils.JortTestCase):
  """Base class for CallTorch tests including numerical checks and boilerplate."""

  def assert_call_torch_convert_and_compare(
      self,
      test_module: Any,
      args: Any,
      atol: float = 1e-5,
      rtol: float = 1e-5,
      onnx_dump_prefix: Optional[str] = None,
      verbose: bool = False,
  ):
    """assert the converted jittable jax function and torch module numerical accuracy."""
    # Get Torch model outputs.
    torch_outputs = test_module(*args)
    torch_outputs = jax.tree_map(
        call_torch.torch_tensor_to_jax_array, torch_outputs
    )
    if isinstance(torch_outputs, np.ndarray):
      torch_outputs = (torch_outputs,)

    # Get JAX model outputs.
    if not isinstance(
        test_module,
        (torch.jit.ScriptModule, torch.jit.ScriptFunction, torch.nn.Module),
    ):
      # Prefer torch.jit.trace over torch.jit.script here.
      # See in-depth discussion here
      # https://ppwwyyxx.com/blog/2022/TorchScript-Tracing-vs-Scripting/
      test_module = torch.jit.trace(
          func=test_module,
          example_inputs=args,
      )
    jax_inputs = jax.tree_map(call_torch.torch_tensor_to_jax_array, args)
    jax_fn, jax_params = call_torch.call_torch(
        model=test_module,
        args=args,
        onnx_dump_prefix=onnx_dump_prefix,
        verbose=verbose,
    )
    jax_fn = jax.jit(jax_fn)
    if verbose:
      logging.info(
          "Exported jax function: jax_params = %s, jax_inputs = %s",
          jax_params,
          jax_inputs,
      )
      lowered = jax_fn.lower(jax_params, jax_inputs)
      logging.info("Exported jax function stablehlo: %s", lowered.as_text())
    jax_outputs = jax_fn(jax_params, jax_inputs)

    # Assert if Torch and JAX model result match each other.
    chex.assert_trees_all_close(
        jax.tree_leaves(torch_outputs),
        jax.tree_leaves(jax_outputs),
        atol=atol,
        rtol=rtol,
    )
    if onnx_dump_prefix:
      onnx_model = onnx.load(os.path.join(onnx_dump_prefix, "model.onnx"))
      self.assert_ort_jort_all_close(onnx_model, jax_inputs)
    return jax_fn, jax_params, jax_inputs, torch_outputs, jax_outputs
