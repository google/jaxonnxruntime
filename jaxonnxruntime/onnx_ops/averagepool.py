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
"""Define ONNX AveragePool operator."""
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
import numpy as np

from .maxpool import MaxPool


@handler.register_op("AveragePool")
class AveragePool(handler.Handler):
  """Implementation of the ONNX AveragePool operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    MaxPool._prepare(node, inputs, onnx_jax_impl)  # pylint: disable=protected-access
    node.attrs_dict["count_include_pad"] = node.attrs.get(
        "count_include_pad", 0
    )
    del node.attrs_dict["storage_order"]

  @classmethod
  def version_11(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_11 AveragePool op."""
    cls._prepare(node, inputs, onnx_averagepool)
    return onnx_averagepool


@functools.partial(
    jit,
    static_argnames=(
        "ceil_mode",
        "strides",
        "pads",
        "dilations",
        "kernel_shape",
        "count_include_pad",
    ),
)
def onnx_averagepool(
    *input_args,
    ceil_mode: int,
    strides: Sequence[int],
    pads: Union[Sequence[tuple[int, int]], str],
    dilations: Sequence[int],
    kernel_shape: Sequence[int],
    count_include_pad: int,
):
  """https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#MaxPool for more details."""
  assert len(input_args) == 1
  x = input_args[0]
  if ceil_mode != 0:
    raise ValueError("ceil_mode = 1 is not implement yet.")

  if count_include_pad == 0:
    one = jnp.ones_like(x, dtype=x.dtype)
    wsizes = lax.reduce_window(one, 0.0, lax.add, kernel_shape, strides, pads)
  else:
    wsizes = np.prod(kernel_shape)

  return (
      lax.reduce_window(
          x,
          jnp.array(0, dtype=x.dtype),
          lax.add,
          kernel_shape,
          strides,
          pads,
          None,
          dilations,
      )
      / wsizes
  )
