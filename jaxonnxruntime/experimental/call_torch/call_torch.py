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
import os
from typing import Any, Callable, Tuple, Union

from absl import logging
import jax
from jaxonnxruntime import call_onnx
import torch

import onnx


class TorchONNXExportError(Exception):
  pass


def torch_tensor_to_np_array(tensor):
  if isinstance(tensor, torch.Tensor):
    return tensor.detach().cpu().numpy()
  else:
    raise ValueError("Input must be a PyTorch tensor.")


def call_torch(
    model: Union[
        torch.nn.Module, torch.jit.ScriptModule, torch.jit.ScriptFunction
    ],
    args: Union[Tuple[Any, ...], torch.Tensor],
    onnx_dump_prefix: str | None = None,
) -> Tuple[Callable[..., Any], Any]:
  """Give a pytorch model and return its equivilent jax function.

  Its API interface should be consistent with
  [`torch.onnx.export`](https://pytorch.org/docs/stable/onnx.html#torch.onnx.export)

  Args:
    model: the torch model to be exported.
    args:  (tuple or torch.Tensor), model inputs args for torch.onnx.export.
    onnx_dump_prefix: The onnx_model debug directory.

  Returns:
    A JAX jittable function can be invoked with JAX pytree arguments.
  """
  file_obj = io.BytesIO()
  if logging.vlog_is_on(3):
    verbose = True
  else:
    verbose = False
  try:
    torch.onnx.export(
        model=model,
        args=args,
        f=file_obj,
        export_params=True,
        verbose=verbose,
        dynamic_axes=None,
        keep_initializers_as_inputs=False,
    )
  except Exception as e:
    raise TorchONNXExportError(
        "torch.onnx.export fails. Please debug torch.onnx.export manually"
        " first."
    ) from e
  if onnx_dump_prefix:
    if not os.path.exists(onnx_dump_prefix):
      os.makedirs(onnx_dump_prefix)
    onnx_model_file = os.path.join(onnx_dump_prefix, "model.onnx")
    with open(onnx_model_file, "wb") as f:
      f.write(file_obj.getvalue())
    logging.info("Saving debug model.onnx to %s", onnx_model_file)
  file_obj.seek(0)
  onnx_model = onnx.load(file_obj)
  jax_args = jax.tree_util.tree_leaves(
      jax.tree_map(torch_tensor_to_np_array, args)
  )
  jax_fn, jax_model_params = call_onnx.call_onnx_model(onnx_model, jax_args)
  return jax_fn, jax_model_params
