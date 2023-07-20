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

"""Define ONNX Trilu operator."""
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


@handler.register_op('Trilu')
class Trilu(handler.Handler):
  """Implementation of the ONNX Trilu operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    node.attrs_dict['upper'] = node.attrs.get('upper', 1)
    if config.jaxort_only_allow_initializers_as_static_args:
      if (
          len(node.inputs) == 1
          or node.inputs[1] not in node.context_graph.initializer_dict
      ):
        raise ValueError(
            "Trilu's `k` is not constant defined by the graph initializers but"
            ' used as a static argument. The function wrapped by `jax.jit` will'
            ' output incorrect results if its value changes in another input.'
        )
      node.attrs_dict['k'] = int(
          node.context_graph.initializer_dict[node.inputs[1]].tolist()[0]
      )
    else:
      node.attrs_dict['k'] = int(inputs[1]) if len(inputs) == 2 else 0

  @classmethod
  def version_14(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_14 Trilu op."""
    cls._prepare(node, inputs, onnx_trilu)
    return onnx_trilu


@functools.partial(jit, static_argnames=('upper', 'k'))
def onnx_trilu(*input_args, k, upper):
  """https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#Trilu for more details."""
  assert len(input_args) == 1 or len(input_args) == 2
  data = input_args[0]
  if upper:
    return jnp.triu(data, k)
  return jnp.tril(data, k)
