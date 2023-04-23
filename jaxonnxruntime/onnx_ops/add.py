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
"""Define ONNX Add operator."""
import functools
import inspect
from collections.abc import Callable, Sequence
from typing import Any

from jax import jit
from jax import numpy as jnp
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node


@handler.register_op("Add")
class Add(handler.Handler):
  """Implementation of the ONNX Add operator."""

  @classmethod
  def _prepare(cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any):
    sig = inspect.signature(onnx_jax_impl)
    kwparams = [param.name for param in sig.parameters.values() if param.kind == inspect.Parameter.KEYWORD_ONLY]
    for name in kwparams:
      node.attrs_dict[name] = node.attrs.get(name, None)

  @classmethod
  def version_14(cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]) -> Callable[..., Any]:
    """ONNX version_14 Add op."""
    cls._prepare(node, inputs, onnx_add)
    return onnx_add


@functools.partial(jit, static_argnames=())
def onnx_add(*input_args):
  """The internal jax impl for onnx Add op."""
  assert len(input_args) == 2
  a, b = input_args
  return jnp.add(a, b)
