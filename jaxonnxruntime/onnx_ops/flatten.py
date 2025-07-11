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

"""Define ONNX Flatten operator."""
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
import numpy as np


@handler.register_op("Flatten")
class Flatten(handler.Handler):
  """Implementation of the ONNX Flatten operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    onnx_ops_utils.update_node_attrs_dict(node, onnx_jax_impl)

  @classmethod
  def version_1(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_1 Flatten op."""
    cls._prepare(node, inputs, onnx_flatten)
    return onnx_flatten

  @classmethod
  def version_11(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_11 Flatten op."""
    cls._prepare(node, inputs, onnx_flatten)
    return onnx_flatten

  @classmethod
  def version_13(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_13 Flatten op."""
    cls._prepare(node, inputs, onnx_flatten)
    return onnx_flatten


@functools.partial(jax.jit, static_argnames="axis")
def onnx_flatten(*input_args, axis):
  """https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#Flatten for more details."""
  axis = 1 if axis is None else axis
  assert len(input_args) == 1
  x = input_args[0]
  dim = len(x.shape)
  assert axis <= dim and axis >= -dim, f"axis should with [{-dim}, {dim}]"
  new_shape = (
      (1, -1) if axis == 0 else (-1, np.prod(x.shape[axis:]).astype(int))
  )
  return jnp.reshape(x, new_shape)
