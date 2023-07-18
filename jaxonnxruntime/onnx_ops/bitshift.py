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

"""Define ONNX BitShift operator."""
# pylint: disable=unused-argument
# pylint: disable=g-explicit-length-test
from collections.abc import Callable, Sequence
import functools
from typing import Any

from jax import jit
from jax import numpy as jnp
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node


@handler.register_op("BitShift")
class BitShift(handler.Handler):
  """Implementation of the ONNX BitShift operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    node.attrs_dict["direction"] = node.attrs.get("direction")
    if node.attrs_dict["direction"] is None:
      raise ValueError("Operator BitShift requires attribute 'direction'!")
    if (
        node.attrs_dict["direction"] != "LEFT"
        and node.attrs_dict["direction"] != "RIGHT"
    ):
      raise ValueError(
          "Operator BitShift only supports LEFT and RIGHT directions!"
      )

  @classmethod
  def version_11(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_11 BitShift op."""
    cls._prepare(node, inputs, onnx_bitshift)
    return onnx_bitshift


@functools.partial(jit, static_argnames="direction")
def onnx_bitshift(x, y, *, direction):
  """https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#BitShift for more details."""
  if direction == "LEFT":
    return jnp.left_shift(x, y)
  return jnp.right_shift(x, y)
