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

"""Define ONNX Abs operator."""
from collections.abc import Callable
from typing import Any

from jax import jit
from jax import lax

from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node

register_op = handler.register_op
Handler = handler.Handler
OnnxNode = onnx_node.OnnxNode


@register_op("Abs")
class Abs(Handler):
  """Implementation of the ONNX Abs operator."""

  @classmethod
  def version_1(cls, node: onnx_node.OnnxNode) -> Callable[..., Any]:
    """Return the absolute value of the input."""
    return onnx_abs

  @classmethod
  def version_6(cls, node: onnx_node.OnnxNode) -> Callable[..., Any]:
    """Return the absolute value of the input."""
    return onnx_abs

  @classmethod
  def version_13(cls, node: onnx_node.OnnxNode) -> Callable[..., Any]:
    """Return the absolute value of the input."""
    return onnx_abs


@jit
def onnx_abs(x):
  """Element-wise absolute value of `x`."""
  return lax.abs(x)
