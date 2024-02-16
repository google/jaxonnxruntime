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

"""Define ONNX Pow operator."""
from collections.abc import Callable, Sequence
import functools
from typing import Any

import jax
from jax import numpy as jnp
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node
from jaxonnxruntime.onnx_ops import onnx_ops_utils


@handler.register_op("Pow")
class Pow(handler.Handler):
  """Implementation of the ONNX Pow operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    onnx_ops_utils.update_node_attrs_dict(node, onnx_jax_impl)

  @classmethod
  def version_1(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_1 Pow op."""
    cls._prepare(node, inputs, onnx_pow)
    return onnx_pow

  @classmethod
  def version_7(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_7 Pow op."""
    cls._prepare(node, inputs, onnx_pow)
    return onnx_pow

  @classmethod
  def version_12(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_12 Pow op."""
    cls._prepare(node, inputs, onnx_pow)
    return onnx_pow

  @classmethod
  def version_13(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_13 Pow op."""
    cls._prepare(node, inputs, onnx_pow)
    return onnx_pow

  @classmethod
  def version_15(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_15 Pow op."""
    cls._prepare(node, inputs, onnx_pow)
    return onnx_pow


@functools.partial(jax.jit, static_argnames=())
def onnx_pow(a, b):
  """The impl for https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#Pow."""
  return jnp.power(a, b).astype(a.dtype)
