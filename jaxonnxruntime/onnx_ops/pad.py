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
"""Define ONNX Pad operator."""
from collections.abc import Callable, Sequence
import functools
from typing import Any

from jax import jit
from jax import numpy as jnp
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node


@handler.register_op('Pad')
class Pad(handler.Handler):
  """Implementation of the ONNX Pad operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    node.attrs_dict['mode'] = node.attrs.get('mode', 'constant')
    node.attrs_dict['pads'] = tuple(inputs[1].tolist())

    if len(inputs) >= 3:
      node.attrs_dict['constant_value'] = inputs[2].item()
    else:
      node.attrs_dict['constant_value'] = 0.0

  @classmethod
  def version_13(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_13 Pad op."""
    cls._prepare(node, inputs, onnx_pad)
    return onnx_pad


@functools.partial(jit, static_argnames=('pads', 'constant_value', 'mode'))
def onnx_pad(*input_args, pads, constant_value=0.0, mode='constant'):
  """The impl for https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#Pad."""
  x = input_args[0]
  input_rank = x.ndim
  if input_rank * 2 != jnp.size(pads):
    raise ValueError(
        'The number of elements in raw_pads should be 2 * input_rank'
        f'pads = {pads}, input_rank = {input_rank}'
    )

  # re-order to np.pad accepted order ((x1_begin, x1_end), (x2_begin, x2_end))
  pad_width = ()
  for i in range(int(jnp.size(pads) / 2)):
    pad_width += (((pads[i], pads[i + input_rank])),)

  if mode == 'constant':
    return jnp.pad(x, pad_width, mode, constant_values=constant_value)
  return jnp.pad(x, pad_width, mode)
