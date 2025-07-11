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

"""Define ONNX Min operator."""
# pylint: disable=unused-argument
# pylint: disable=g-explicit-length-test
from collections.abc import Callable, Sequence
import functools
from typing import Any

import jax
from jax import numpy as jnp
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node
from jaxonnxruntime.onnx_ops import onnx_ops_utils


@handler.register_op("Min")
class Min(handler.Handler):
  """Implementation of the ONNX Min operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    onnx_ops_utils.update_node_attrs_dict(node, onnx_jax_impl)
    node.attrs_dict["arg_num"] = len(node.inputs)

  @classmethod
  def version_6(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_6 Min op."""
    cls._prepare(node, inputs, onnx_min)
    return onnx_min

  @classmethod
  def version_8(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_8 Min op."""
    cls._prepare(node, inputs, onnx_min)
    return onnx_min

  @classmethod
  def version_12(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_12 Min op."""
    cls._prepare(node, inputs, onnx_min)
    return onnx_min

  @classmethod
  def version_13(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_13 Min op."""
    cls._prepare(node, inputs, onnx_min)
    return onnx_min


@functools.partial(jax.jit, static_argnames=("arg_num",))
def onnx_min(*input_args, arg_num):
  """https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#Min for more details."""
  assert len(input_args) == arg_num
  res = input_args[0]
  for i in range(arg_num):
    res = jnp.minimum(res, input_args[i])
  return res
