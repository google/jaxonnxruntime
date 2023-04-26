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
"""Define ONNX ReduceMax operator."""
import functools
import inspect
from collections.abc import Callable, Sequence
from typing import Any

from jax import jit
from jax import numpy as jnp
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node


@handler.register_op("ReduceMax")
class ReduceMax(handler.Handler):
  """Implementation of the ONNX ReduceMax operator."""

  @classmethod
  def _prepare(cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any):
    sig = inspect.signature(onnx_jax_impl)
    kwparams = [param.name for param in sig.parameters.values() if param.kind == inspect.Parameter.KEYWORD_ONLY]
    for name in kwparams:
      node.attrs_dict[name] = node.attrs.get(name, None)
    node.attrs_dict['keepdims'] = True if node.attrs_dict['keepdims'] == 1 else False

  @classmethod
  def version_13(cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]) -> Callable[..., Any]:
    """ONNX version_13 ReduceMax op."""
    cls._prepare(node, inputs, onnx_reducemax)
    return onnx_reducemax


@functools.partial(jit, static_argnames=('axes', 'keepdims'))
def onnx_reducemax(*input_args, axes, keepdims=True):
  """The impl for https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#ReduceMax."""
  assert len(input_args) == 1
  x = input_args[0]
  return jnp.max(x, axis=axes, keepdims=keepdims)