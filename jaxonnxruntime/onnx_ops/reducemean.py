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
"""Define ONNX ReduceMean operator."""
from collections.abc import Callable, Sequence
import functools
import inspect
from typing import Any

from jax import jit
from jax import numpy as jnp
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node


@handler.register_op('ReduceMean')
class ReduceMean(handler.Handler):
  """Implementation of the ONNX ReduceMean operator."""

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
    node.attrs_dict['keepdims'] = node.attrs.get('keepdims', True)

  @classmethod
  def version_13(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_13 ReduceMean op."""
    cls._prepare(node, inputs, onnx_reducemean)
    return onnx_reducemean


@functools.partial(jit, static_argnames=('axes', 'keepdims'))
def onnx_reducemean(*input_args, axes=None, keepdims=True):
  """The impl for https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#ReduceMean."""
  assert len(input_args) == 1
  data = input_args[0]
  return jnp.mean(data, axis=axes, keepdims=keepdims)
