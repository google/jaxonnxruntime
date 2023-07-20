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

"""Define ONNX GatherElements operator."""
# pylint: disable=unused-argument
# pylint: disable=g-explicit-length-test
from collections.abc import Callable, Sequence
import functools
from typing import Any

from jax import jit
from jax import numpy as jnp
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node


@handler.register_op('GatherElements')
class GatherElements(handler.Handler):
  """Implementation of the ONNX GatherElements operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    node.attrs_dict['axis'] = node.attrs.get('axis', 0)

  @classmethod
  def version_13(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_13 GatherElements op."""
    cls._prepare(node, inputs, onnx_gatherelements)
    return onnx_gatherelements


@functools.partial(jit, static_argnames=('axis',))
def onnx_gatherelements(*input_args, axis):
  """https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#GatherElements for more details."""
  data, index = input_args
  data_swaped = jnp.swapaxes(data, 0, axis)
  index_swaped = jnp.swapaxes(index, 0, axis)
  gathered = jnp.choose(index_swaped, data_swaped, mode='wrap')
  return jnp.swapaxes(gathered, 0, axis)
