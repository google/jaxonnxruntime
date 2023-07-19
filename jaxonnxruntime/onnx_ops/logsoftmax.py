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

"""Define ONNX LogSoftmax operator."""
# pylint: disable=unused-argument
# pylint: disable=g-explicit-length-test
from collections.abc import Callable, Sequence
import functools
from typing import Any

import jax
from jax import jit
from jax import numpy as jnp
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node


@handler.register_op('LogSoftmax')
class LogSoftmax(handler.Handler):
  """Implementation of the ONNX LogSoftmax operator."""

  @classmethod
  def _prepare_1(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    node.attrs_dict['axis'] = node.attrs.get('axis', 1)

  @classmethod
  def _prepare_11(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    cls._prepare_1(node, inputs, onnx_jax_impl)

  @classmethod
  def _prepare_13(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    node.attrs_dict['axis'] = node.attrs.get('axis', -1)

  @classmethod
  def version_1(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_1 LogSoftmax op."""
    cls._prepare_1(node, inputs, onnx_logsoftmax)
    return onnx_logsoftmax

  @classmethod
  def version_11(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_11 LogSoftmax op."""
    cls._prepare_11(node, inputs, onnx_logsoftmax)
    return onnx_logsoftmax

  @classmethod
  def version_13(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_13 LogSoftmax op."""
    cls._prepare_13(node, inputs, onnx_logsoftmax)
    return onnx_logsoftmax


@functools.partial(jit, static_argnames=('axis',))
def onnx_logsoftmax(*input_args, axis=-1):
  """https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#LogSoftmax for more details."""
  assert len(input_args) == 1
  data = input_args[0]
  res = jax.nn.softmax(data, axis=axis)
  return jnp.log(res)
