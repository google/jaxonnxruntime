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
"""Define ONNX Reshape operator."""
from collections.abc import Callable, Sequence
import functools
from typing import Any
from jax import jit
from jax import numpy as jnp
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node
import numpy as np


@handler.register_op('Reshape')
class Reshape(handler.Handler):
  """Implementation of the ONNX Reshape operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    node.attrs_dict['shape'] = tuple(inputs[1].tolist())
    node.attrs_dict['allowzero'] = node.attrs.get('allowzero', 0)

  @classmethod
  def version_14(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_14 Reshape op."""
    cls._prepare(node, inputs, onnx_reshape)
    return onnx_reshape


@functools.partial(jit, static_argnames=('shape', 'allowzero'))
def onnx_reshape(*input_args, shape, allowzero):
  """The impl for https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#Reshape."""
  assert len(input_args) == 2
  data, _ = input_args
  new_shape = np.copy(shape)
  if allowzero == 0:
    zeros_index = np.where(np.array(shape) == 0)
    new_shape[zeros_index] = np.array(data.shape)[zeros_index]
  return jnp.reshape(data, new_shape)
