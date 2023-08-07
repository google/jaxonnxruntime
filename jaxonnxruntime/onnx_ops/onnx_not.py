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

"""Define ONNX Not operator."""
from collections.abc import Callable, Sequence
import functools
from typing import Any

from jax import jit
from jax import numpy as jnp
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node


@handler.register_op("Not")
class Not(handler.Handler):
  """Implementation of the ONNX Not operator."""

  @classmethod
  def version_1(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_1 Not op."""
    return onnx_not


@functools.partial(jit, static_argnames=())
def onnx_not(*input_args):
  """The impl for https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#Not."""
  assert len(input_args) == 1
  x = input_args[0]
  return (jnp.logical_not(x),)
