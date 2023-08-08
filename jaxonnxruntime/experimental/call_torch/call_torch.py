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

"""Convert PyTorch function to Jax funtion."""

import io
from typing import Any, Callable, Tuple, Union

from absl import logging
import jax
from jaxonnxruntime import call_onnx
import torch

import onnx


def torch_tensor_to_np_array(tensor):
  if isinstance(tensor, torch.Tensor):
    return tensor.detach().cpu().numpy()
  else:
    raise ValueError("Input must be a PyTorch tensor.")


def call_torch(
    model: Union[torch.jit.ScriptModule, torch.jit.ScriptFunction],
    args: Union[Tuple[Any, ...], torch.Tensor],
) -> Tuple[Callable[..., Any], Any]:
  """Give a pytorch model and return its equivilent jax function.

  Its API interface should be consistent with
  [`torch.onnx.export`](https://pytorch.org/docs/stable/onnx.html#torch.onnx.export)

  Args:
    model: the torchs_cript model to be exported.
    args:  (tuple or torch.Tensor), model inputs args for torch.onnx.export.

  Returns:
    A JAX jittable function can be invoked with JAX pytree arguments.
  """
  file_obj = io.BytesIO()
  if logging.vlog_is_on(3):
    verbose = True
  else:
    verbose = False
  torch.onnx.export(
      model=model,
      args=args,
      f=file_obj,
      export_params=True,
      verbose=verbose,
      dynamic_axes=None,
      keep_initializers_as_inputs=False,
  )
  file_obj.seek(0)
  onnx_model = onnx.load(file_obj)
  jax_args = jax.tree_util.tree_leaves(
      jax.tree_map(torch_tensor_to_np_array, args)
  )
  jax_fn, jax_model_params = call_onnx.call_onnx_model(onnx_model, jax_args)
  return jax_fn, jax_model_params
