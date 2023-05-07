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
"""Define ONNX MaxPool operator."""
# pylint: disable=unused-argument
# pylint: disable=g-explicit-length-test

from collections.abc import Callable, Sequence
import functools
from typing import Any, Union

from jax import jit
from jax import lax
from jax import numpy as jnp
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node


@handler.register_op("MaxPool")
class MaxPool(handler.Handler):
  """Implementation of the ONNX MaxPool operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    node.attrs_dict["ceil_mode"] = node.attrs.get("ceil_mode", 0)
    node.attrs_dict["storage_order"] = node.attrs.get("storage_order", 0)
    strides = node.attrs.get("strides", None)
    x = inputs[0]
    node.attrs_dict["strides"] = (
        ((1,) * (x.ndim - len(strides)) + tuple(strides))
        if strides
        else (1,) * x.ndim
    )
    dilations = node.attrs.get("dilations", None)
    node.attrs_dict["dilations"] = (
        ((1,) * (x.ndim - len(dilations)) + tuple(dilations))
        if dilations
        else (1,) * x.ndim
    )
    kernel_shape = node.attrs.get("kernel_shape", None)
    node.attrs_dict["kernel_shape"] = (1,) * (
        x.ndim - len(kernel_shape)
    ) + tuple(kernel_shape)

    if "pads" in node.attrs:
      pads = node.attrs["pads"]
      # ONNX follows [x1_begin, x2_begin...x1_end, x2_end,...].
      # lax conv is a sequence of n (low, high) integer pairs.
      n = len(pads) // 2

      pads_new = [(0, 0) for i in range(inputs[0].ndim - 2)] + [
          (pads[i], pads[i + n]) for i in range(n)
      ]
      node.attrs_dict["pads"] = tuple(pads_new)
    else:
      pad_str_type = node.attrs.get("auto_pad", "VALID")
      onnx_to_jax_pad_type = {
          "SAME_UPPER": "SAME",
          "VALID": "VALID",
          "SAME_LOWER": "SAME_LOWER",
      }
      assert (
          pad_str_type in onnx_to_jax_pad_type
      ), f"Invalid auto_pad attribute: {pad_str_type}"
      node.attrs_dict["pads"] = onnx_to_jax_pad_type[pad_str_type]

  @classmethod
  def version_12(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_12 MaxPool op."""
    cls._prepare(node, inputs, onnx_maxpool)
    return onnx_maxpool


@functools.partial(
    jit,
    static_argnames=(
        "ceil_mode",
        "strides",
        "pads",
        "dilations",
        "kernel_shape",
    ),
)
def onnx_maxpool(
    *input_args,
    ceil_mode: int,
    strides: Sequence[int],
    pads: Union[Sequence[tuple[int, int]], str],
    dilations: Sequence[int],
    kernel_shape: Sequence[int],
    storage_order: int,
):
  """https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#MaxPool for more details."""
  assert len(input_args) == 1
  x = input_args[0]
  if ceil_mode != 0:
    raise ValueError("ceil_mode = 1 is not implement yet.")

  return lax.reduce_window(
      x, -jnp.inf, lax.max, kernel_shape, strides, pads, None, dilations
  )
