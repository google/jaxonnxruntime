# Copyright 2025 The Jaxonnxruntime Authors.
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

"""Define ONNX Tile operator."""

# pylint: disable=unused-argument
# pylint: disable=g-explicit-length-test
from collections.abc import Callable, Sequence
import functools
from typing import Any

import jax
from jax import numpy as jnp
from jaxonnxruntime.core import config_class

config = config_class.config
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node


@handler.register_op('Tile')
class Tile(handler.Handler):
  """Implementation of the ONNX Tile operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    if config.jaxort_only_allow_initializers_as_static_args:
      if node.inputs[1] not in node.context_graph.get_constant_dict():
        raise ValueError(
            f'{node.inputs[1]} is not constant but used as `repeats` of Tile'
            ' static argument during `jax.jit`. the jitted function gives'
            ' wrong results if its value changes in another input.If you know'
            ' what you are doing, set'
            ' `config.update("jaxort_only_allow_initializers_as_static_args",'
            ' False)` to remove this contraint.'
        )
      node.attrs_dict['repeats'] = tuple(
          node.context_graph.get_constant_dict()[node.inputs[1]].tolist()
      )
    else:
      node.attrs_dict['repeats'] = tuple(inputs[1].tolist())

  @classmethod
  def version_1(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_1 Tile op."""
    cls._prepare(node, inputs, onnx_tile)
    return onnx_tile

  @classmethod
  def version_6(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_6 Tile op."""
    cls._prepare(node, inputs, onnx_tile)
    return onnx_tile

  @classmethod
  def version_13(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_13 Tile op."""
    cls._prepare(node, inputs, onnx_tile)
    return onnx_tile


@functools.partial(jax.jit, static_argnames='repeats')
def onnx_tile(*input_args, repeats):
  """https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#Tile for more details."""
  assert (
      len(input_args) == 2
  ), f'Expected 2 input args but got {len(input_args)}'
  x = input_args[0]
  return jnp.tile(x, repeats)
