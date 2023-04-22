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
"""Define ONNX Split operator."""
from collections.abc import Callable, Sequence
import functools
from typing import Any

from jax import jit
from jax import lax
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node


@handler.register_op('Split')
class Split(handler.Handler):
  """Implementation of the ONNX Split operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    node.attrs_dict['axis'] = (
        0 if 'axis' not in node.attrs else node.attrs['axis']
    )
    node.attrs_dict['num_output'] = node.len_outputs
    if len(inputs) >= 2:
      node.attrs_dict['split'] = tuple(inputs[1].tolist())

  @classmethod
  def version_13(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_13 Split op."""
    cls._prepare(node, inputs, onnx_split)
    return onnx_split


@functools.partial(jit, static_argnames=('split', 'axis', 'num_output'))
def onnx_split(*input_args, num_output, split=None, axis=0):
  """https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#Split for more details."""

  x = input_args[0]
  if split is None:
    split = [x.shape[axis] // num_output] * num_output
  starts = []
  ends = []
  starts.append([0] * x.ndim)
  for idx in range(1, num_output):
    st = [0] * x.ndim
    st[axis] = sum(split[:idx])
    starts.append(st)
    en = list(x.shape)
    en[axis] = sum(split[:idx])
    ends.append(en)
  ends.append(list(x.shape))

  return [lax.slice(x, start, end) for start, end in zip(starts, ends)]
