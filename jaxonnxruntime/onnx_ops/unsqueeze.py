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
"""Define ONNX Unsqueeze operator."""
from collections.abc import Callable, Sequence
import functools
from typing import Any

from jax import jit
from jax import numpy as jnp
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node


@handler.register_op('Unsqueeze')
class Unsqueeze(handler.Handler):
  """Implementation of the ONNX Unsqueeze operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    if 'axes' in node.attrs:
      node.attrs_dict['axis'] = node.attrs['axes']
    if len(inputs) >= 2:
      node.attrs_dict['axis'] = tuple(inputs[1].tolist())

  @classmethod
  def version_13(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_13 Unsqueeze op."""
    cls._prepare(node, inputs, onnx_unsqueeze)
    return onnx_unsqueeze


@functools.partial(jit, static_argnames='axis')
def onnx_unsqueeze(*input_args, axis: list[int]):
  """The impl for https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#Unsqueeze."""
  x = input_args[0]
  return jnp.expand_dims(x, axis)
