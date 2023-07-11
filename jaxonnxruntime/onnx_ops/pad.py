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
from jaxonnxruntime import config
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node


@handler.register_op('Pad')
class Pad(handler.Handler):
  """Implementation of the ONNX Pad operator."""

  @classmethod
  def _prepare_2(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    node.attrs_dict['mode'] = node.attrs.get('mode', 'constant')
    node.attrs_dict['pads'] = tuple(node.attrs.get('pads'))
    node.attrs_dict['constant_value'] = node.attrs.get('value', 0.0)
    node.attrs_dict['axes'] = tuple(range(len(inputs[0].shape)))

  @classmethod
  def _prepare_13(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    node.attrs_dict['mode'] = node.attrs.get('mode', 'constant')

    if config.jaxort_only_allow_initializers_as_static_args:
      for index in range(1, len(node.inputs)):
        if node.inputs[index] not in node.context_graph.initializer_dict:
          raise ValueError(
              f'{node.inputs[index]} is not constant but used as `pads`'
              ' static argument during `jax.jit`. '
              'the jitted function gives wrong results if its value changes'
              'in another input.'
          )
    if len(inputs) >= 2:
      node.attrs_dict['pads'] = tuple(inputs[1].tolist())

    if len(inputs) >= 3:
      node.attrs_dict['constant_value'] = inputs[2].item()
    else:
      node.attrs_dict['constant_value'] = 0.0

    if len(inputs) >= 4:
      node.attrs_dict['axes'] = tuple(inputs[3].tolist())
    else:
      node.attrs_dict['axes'] = tuple(range(len(inputs[0].shape)))

  @classmethod
  def _prepare_19(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    cls._prepare_13(node, inputs, onnx_jax_impl)

  @classmethod
  def version_2(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_2 Pad op."""
    cls._prepare_2(node, inputs, onnx_pad)
    return onnx_pad

  @classmethod
  def version_13(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_13 Pad op."""
    cls._prepare_13(node, inputs, onnx_pad)
    return onnx_pad

  @classmethod
  def version_19(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_19 Pad op."""
    cls._prepare_19(node, inputs, onnx_pad)
    return onnx_pad


@functools.partial(
    jit, static_argnames=('pads', 'constant_value', 'mode', 'axes')
)
def onnx_pad(
    *input_args,
    pads: Sequence[int],
    constant_value: Any,
    mode: str,
    axes: Sequence[int],
):
  """The impl for https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#Pad."""
  x = input_args[0]
  input_rank = x.ndim
  axes = [axis if axis >= 0 else axis + input_rank for axis in axes]
  num_axes = len(axes)
  if num_axes * 2 != jnp.size(pads):
    raise ValueError(
        'The number of elements in raw_pads should be 2 * len(axis)'
        f'pads = {pads}, axis = {axes}'
    )

  # re-order to np.pad accepted order ((x1_begin, x1_end), (x2_begin, x2_end))
  pad_width = []
  for _ in range(input_rank):
    pad_width += [[0, 0]]

  for i in range(num_axes):
    axis = axes[i]
    pad_width[axis] = [pads[i], pads[i + num_axes]]

  if mode == 'constant':
    return jnp.pad(x, pad_width, mode, constant_values=constant_value)
  return jnp.pad(x, pad_width, mode)
