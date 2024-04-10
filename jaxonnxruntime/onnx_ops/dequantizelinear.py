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

"""Define ONNX DequantizeLinear operator."""
# pylint: disable=unused-argument
# pylint: disable=g-explicit-length-test
from collections.abc import Callable, Sequence
import functools
from typing import Any, Optional, Tuple

import jax
from jax import numpy as jnp
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node
from jaxonnxruntime.onnx_ops import onnx_ops_utils


@handler.register_op("DequantizeLinear")
class DequantizeLinear(handler.Handler):
  """Implementation of the ONNX DequantizeLinear operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    onnx_node.update_node_attr_dict_with_jax_func_kwargs(node, onnx_jax_impl)
    onnx_ops_utils.update_node_attrs_dict(node, onnx_jax_impl)

  @classmethod
  def version_10(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_10 DequantizeLinear op."""
    cls._prepare(node, inputs, onnx_dequantizelinear)
    return onnx_dequantizelinear

  @classmethod
  def version_13(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_13 DequantizeLinear op."""
    cls._prepare(node, inputs, onnx_dequantizelinear)
    return onnx_dequantizelinear

  @classmethod
  def version_19(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_19 DequantizeLinear op."""
    cls._prepare(node, inputs, onnx_dequantizelinear)
    return onnx_dequantizelinear

def reshape_input(
    value: jnp.ndarray, shape: Tuple[int, ...], axis: Optional[int]
) -> jnp.ndarray:
  """Reshape input array to the given shape and axis."""
  assert axis is not None
  if len(value.shape) == 0:
    return value
  dims = [1] * len(shape)
  try:
    dims[axis] = value.size
  except IndexError as e:
    raise IndexError(
        f"axis is out of boundary, axis={axis}, "
        f"value.shape={value.shape}, shape={shape}."
    ) from e
  return value.reshape(tuple(dims))


@functools.partial(jax.jit, static_argnames="axis")
def onnx_dequantizelinear(*input_args, axis=None):
  """https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#DequantizeLinear for more details."""
  x, x_scale, x_zero_point = onnx_node.pad_sequence(
      input_args, 3, pad_value=jnp.zeros_like(input_args[1])
  )
  axis = 1 if axis is None else axis
  x_scale_dtype = x_scale.dtype
  assert jax.dtypes.issubdtype(
      x_scale_dtype, jnp.float32
  ) or jax.dtypes.issubdtype(x_scale_dtype, jnp.float64), x_scale_dtype
  dx = x.astype(jnp.float32) - reshape_input(x_zero_point, x.shape, axis)
  y = dx * reshape_input(x_scale, x.shape, axis)
  return y
