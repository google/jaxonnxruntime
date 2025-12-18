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

"""Define ONNX ScatterElements operator."""

from collections.abc import Callable, Sequence
import functools
from typing import Any

import jax
from jax import numpy as jnp
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node


@handler.register_op("ScatterElements")
class ScatterElements(handler.Handler):
  """Implementation of the ONNX ScatterElements operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    node.attrs_dict["axis"] = node.attrs.get("axis", 0)

    reduction_map = {
        None: "set",
        "add": "add",
        "mul": "multiply",
        "max": "max",
        "min": "min",
    }
    node.attrs_dict["reduction"] = reduction_map[
        node.attrs.get("reduction", None)
    ]

  @classmethod
  def version_11(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_11 ScatterElements op."""
    cls._prepare(node, inputs, onnx_scatterelements)
    return onnx_scatterelements

  @classmethod
  def version_13(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_13 ScatterElements op."""
    cls._prepare(node, inputs, onnx_scatterelements)
    return onnx_scatterelements

  @classmethod
  def version_16(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_16 ScatterElements op."""
    cls._prepare(node, inputs, onnx_scatterelements)
    return onnx_scatterelements

  @classmethod
  def version_18(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_18 ScatterElements op."""
    cls._prepare(node, inputs, onnx_scatterelements)
    return onnx_scatterelements

@functools.partial(jax.jit, static_argnames=("axis", "reduction"))
def onnx_scatterelements(*input_args, axis, reduction):
  """https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#ScatterElements for more details."""
  data, indices, updates = input_args

  idx = list(
      jnp.meshgrid(
          *(jnp.arange(n) for n in data.shape), sparse=True, indexing="ij"
      )
  )
  idx[axis] = indices
  out = getattr(data.at[tuple(idx)], reduction)(
      updates, indices_are_sorted=False, unique_indices=False
  )
  return out
