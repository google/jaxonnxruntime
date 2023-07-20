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

"""Define ONNX TopK operator."""
# pylint: disable=unused-argument
# pylint: disable=g-explicit-length-test
from collections.abc import Callable, Sequence
import functools
from typing import Any

from jax import jit
from jax import numpy as jnp
from jaxonnxruntime import config
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node


@handler.register_op('TopK')
class TopK(handler.Handler):
  """Implementation of the ONNX TopK operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    if config.jaxort_only_allow_initializers_as_static_args:
      if node.inputs[1] not in node.context_graph.initializer_dict:
        raise ValueError(
            f'{node.inputs[1]} is not constant defined by the graph'
            " initializers but used as TopK's static argument `k`. The"
            ' function wrapped by `jax.jit` will output incorrect results if'
            ' its value changes in another input.'
        )
      node.attrs_dict['k'] = int(
          node.context_graph.initializer_dict[node.inputs[1]].tolist()[0]
      )
    else:
      node.attrs_dict['k'] = int(inputs[1].tolist()[0])
    node.attrs_dict['axis'] = node.attrs.get('axis', -1)
    node.attrs_dict['largest'] = node.attrs.get('largest', 1)
    node.attrs_dict['sorted'] = node.attrs.get('sorted', 1)

  @classmethod
  def version_11(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_11 TopK op."""
    cls._prepare(node, inputs, onnx_topk)
    return onnx_topk


@functools.partial(jit, static_argnames=('k', 'axis', 'largest', 'sorted'))
def onnx_topk(*input_args, k, axis, largest, sorted):
  """https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#TopK for more details."""
  assert len(input_args) == 2
  data, _ = input_args
  axis = axis if axis >= 0 else (axis + len(data.shape))
  sort, sorti = topk_sorted_implementation(data, k, axis, largest)
  return (sort, sorti.astype(jnp.int64))


def topk_sorted_implementation(data, k, axis, largest):
  """Topk implementation."""
  if len(data.shape) == 2 and axis == 1:
    sample_range = jnp.arange(data.shape[0])[:, None]
    if largest == 0:
      sorted_indices = jnp.argpartition(data, axis=axis, kth=k - 1)
      sorted_indices = sorted_indices[:, :k]
      # argpartition doesn't guarantee sorted order, so we sort again
      sorted_indices = sorted_indices[
          sample_range, jnp.argsort(data[sample_range, sorted_indices])
      ]
    else:
      sorted_indices = jnp.argpartition(-data, axis=axis, kth=k - 1)
      sorted_indices = sorted_indices[:, :k]
      # argpartition doesn't guarantee sorted order, so we sort again
      sorted_indices = sorted_indices[
          sample_range, jnp.argsort(-data[sample_range, sorted_indices])
      ]
    sorted_distances = data[sample_range, sorted_indices]
    return sorted_distances, sorted_indices

  sorted_indices = jnp.argsort(data, axis=axis)
  sorted_values = jnp.sort(data, axis=axis)
  if largest:
    sorted_indices = jnp.flip(sorted_indices, axis=axis)
    sorted_values = jnp.flip(sorted_values, axis=axis)
  ark = jnp.arange(k)
  topk_sorted_indices = jnp.take(sorted_indices, ark, axis=axis)
  topk_sorted_values = jnp.take(sorted_values, ark, axis=axis)
  return topk_sorted_values, topk_sorted_indices
