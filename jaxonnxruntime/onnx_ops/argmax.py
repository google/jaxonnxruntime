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

"""Define ONNX ArgMax operator."""
# pylint: disable=unused-argument
# pylint: disable=g-explicit-length-test
from collections.abc import Callable, Sequence
import functools
from typing import Any

from jax import jit
from jax import numpy as jnp
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node


@handler.register_op('ArgMax')
class ArgMax(handler.Handler):
  """Implementation of the ONNX ArgMax operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    node.attrs_dict['axis'] = node.attrs.get('axis', 0)
    node.attrs_dict['keepdims'] = node.attrs.get('keepdims', 1)
    node.attrs_dict['select_last_index'] = node.attrs.get(
        'select_last_index', 0
    )

  @classmethod
  def version_13(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_13 ArgMax op."""
    cls._prepare(node, inputs, onnx_argmax)
    return onnx_argmax


@functools.partial(
    jit, static_argnames=('axis', 'keepdims', 'select_last_index')
)
def onnx_argmax(data, *, axis, keepdims, select_last_index):
  """https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#ArgMax for more details."""
  keepdims = False if keepdims == 0 else True
  if select_last_index == 0:
    return jnp.argmax(data, axis=axis, keepdims=keepdims)
  data = jnp.flip(data, axis)
  result = jnp.argmax(data, axis=axis)
  result = data.shape[axis] - result - 1
  if keepdims:
    result = jnp.expand_dims(result, axis)
  return result
