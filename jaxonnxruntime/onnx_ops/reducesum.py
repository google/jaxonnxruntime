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
"""Define ONNX ReduceSum operator."""
# pylint: disable=g-explicit-length-test
from collections.abc import Callable, Sequence
import inspect
from typing import Any

from jax import numpy as jnp
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node


@handler.register_op('ReduceSum')
class ReduceSum(handler.Handler):
  """Implementation of the ONNX ReduceSum operator."""

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

    if len(inputs) >= 2:
      node.attrs_dict['axes'] = tuple(inputs[1].tolist())
      node.attrs_dict['axes'] = (
          None if len(node.attrs_dict['axes']) == 0 else node.attrs_dict['axes']
      )
    node.attrs_dict['keepdims'] = (
        True if node.attrs_dict['keepdims'] == 1 else False
    )

  @classmethod
  def version_13(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_13 ReduceSum op."""
    cls._prepare(node, inputs, onnx_reducesum)
    return onnx_reducesum


# @functools.partial(jit, static_argnames=('axes', 'keepdims'))
def onnx_reducesum(
    *input_args, axes=None, keepdims=False, noop_with_empty_axes=None
):
  """The impl for https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#ReduceSum."""
  assert len(input_args) == 1 or len(input_args) == 2
  data = input_args[0]
  noop_with_empty_axes = 0 if not noop_with_empty_axes else noop_with_empty_axes
  if noop_with_empty_axes != 0:
    return data
  return jnp.sum(data, axis=axes, keepdims=keepdims)
