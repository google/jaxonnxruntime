# Copyright 2026 The Jaxonnxruntime Authors.
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

"""Define ONNX Clip operator."""
# pylint: disable=unused-argument
# pylint: disable=g-explicit-length-test
from collections.abc import Callable, Sequence
import functools
from typing import Any

import jax
from jax import numpy as jnp
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node


@handler.register_op('Clip')
class Clip(handler.Handler):
  """Implementation of the ONNX Clip operator."""

  @classmethod
  def _prepare_6(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ) -> None:
    node.attrs_dict['amin'] = node.attrs.get('min')
    node.attrs_dict['amax'] = node.attrs.get('max')

  @classmethod
  def _prepare_13(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ) -> None:
    pass

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ) -> None:
    cls._prepare_13(node, inputs, onnx_jax_impl)

  @classmethod
  def version_6(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_6 Clip op."""
    cls._prepare_6(node, inputs, onnx_clip)
    return onnx_clip

  @classmethod
  def version_11(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_11 Clip op."""
    cls._prepare_13(node, inputs, onnx_clip)
    return onnx_clip

  @classmethod
  def version_13(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_13 Clip op."""
    cls._prepare_13(node, inputs, onnx_clip)
    return onnx_clip


@functools.partial(jax.jit, static_argnames=())
def onnx_clip(*input_args: jax.Array, amin=None, amax=None) -> jax.Array:
  """The impl for Clip operator."""
  data = input_args[0]
  if len(input_args) >= 2:
    amin = input_args[1]
  if len(input_args) >= 3:
    amax = input_args[2]

  if amin is None and amax is None:
    return data
  return jnp.clip(data, a_min=amin, a_max=amax)
