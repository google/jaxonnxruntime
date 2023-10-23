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

"""Define ONNX QuantizeLinear operator."""
# pylint: disable=unused-argument
# pylint: disable=g-explicit-length-test
from collections.abc import Callable, Sequence
import functools
import inspect
from typing import Any

import jax
from jax import numpy as jnp
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node
from jaxonnxruntime.core import onnx_utils

import onnx


@handler.register_op("QuantizeLinear")
class QuantizeLinear(handler.Handler):
  """Implementation of the ONNX QuantizeLinear operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    sig = inspect.signature(onnx_jax_impl)
    kwparams = [
        param.name
        for param in sig.parameters.values()
        if param.kind == inspect.Parameter.KEYWORD_ONLY
    ]
    for name in kwparams:
      node.attrs_dict[name] = node.attrs.get(name, None)

  @classmethod
  def version_10(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_10 QuantizeLinear op."""
    cls._prepare(node, inputs, onnx_quantizelinear)
    return onnx_quantizelinear

  @classmethod
  def version_13(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_13 QuantizeLinear op."""
    cls._prepare(node, inputs, onnx_quantizelinear)
    return onnx_quantizelinear

  @classmethod
  def version_19(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_19 QuantizeLinear op."""
    cls._prepare(node, inputs, onnx_quantizelinear)
    return onnx_quantizelinear


@functools.partial(jax.jit, static_argnames=("axis", "saturate"))
def onnx_quantizelinear(*input_args, axis, saturate):
  """https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#QuantizeLinear for more details."""
  x, y_scale, zero_point = input_args
  if not axis:
    axis = 1
  if not saturate:
    saturate = True

  if len(y_scale.shape) > 1:
    raise RuntimeError("Input 2 must be a vector or a number.")
  if len(y_scale.shape) > 0 and y_scale.size == 1:
    y_scale = y_scale[0]
  if len(y_scale.shape) > 0:
    new_shape = [1 for _ in x.shape]
    new_shape[axis] = len(y_scale)
    x = x / y_scale.reshape(new_shape)
  else:
    x = x / y_scale
    new_shape = x.shape

  if zero_point is not None:
    tensor_type = onnx_utils.np_dtype_to_tensor_dtype(zero_point.dtype)

    if tensor_type == onnx.TensorProto.UINT8:
      xi = jnp.rint(x).astype(jnp.int32)
      if len(y_scale.shape) > 0:
        xi += zero_point.reshape(new_shape)
      else:
        xi += zero_point
      dtype = onnx_utils.tensor_dtype_to_jnp_dtype(tensor_type)
      return (jnp.clip(xi, 0, 255).astype(dtype),)
    elif tensor_type == onnx.TensorProto.INT8:
      xi = jnp.rint(x).astype(jnp.int32)
      if len(y_scale.shape) > 0:
        xi += zero_point.reshape(new_shape)
      else:
        xi += zero_point
      dtype = onnx_utils.tensor_dtype_to_jnp_dtype(tensor_type)
      return (jnp.clip(xi, -128, 127).astype(dtype),)
    else:
      raise RuntimeError(
          "Currently QuantizeLinear implementation does not support dtype"
          f" {tensor_type}.zero_point.dtype={zero_point.dtype}."
      )
