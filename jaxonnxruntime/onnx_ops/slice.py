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
"""Define ONNX Slice operator."""
from collections.abc import Callable, Sequence
from typing import Any

from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node


@handler.register_op('Slice')
class Slice(handler.Handler):
  """Implementation of the ONNX Slice operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    node.attrs_dict['starts'] = tuple(inputs[1].tolist())
    node.attrs_dict['ends'] = tuple(inputs[2].tolist())
    if len(inputs) >= 4:
      node.attrs_dict['axes'] = tuple(inputs[3].tolist())
    else:
      node.attrs_dict['axes'] = None
    if len(inputs) >= 5:
      node.attrs_dict['steps'] = tuple(inputs[4].tolist())
    else:
      node.attrs_dict['steps'] = None

  @classmethod
  def version_13(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_13 Slice op."""
    cls._prepare(node, inputs, onnx_slice)
    return onnx_slice


# @functools.partial(jit, static_argnames=('starts', 'ends', 'axes', 'steps'))
def onnx_slice(*input_args, starts, ends, axes, steps):
  """The impl for https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#Slice."""
  x = input_args[0]
  if axes is None:
    axes = tuple(range(len(starts)))
  if steps is None:
    steps = [1] * len(starts)
  slices = tuple(
      slice(start, end, step) for start, end, step in zip(starts, ends, steps)
  )
  sub_indx = [slice(None)] * len(x.shape)
  for i, axis in enumerate(axes):
    sub_indx[axis] = slices[i]
  return x[tuple(sub_indx)]
