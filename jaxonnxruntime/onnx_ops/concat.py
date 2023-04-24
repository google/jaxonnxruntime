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
"""Define ONNX Concat operator."""
import functools
from collections.abc import Callable
from typing import Any

from jax import jit
from jax import numpy as jnp
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node


@handler.register_op("Concat")
class Concat(handler.Handler):
  """Implementation of the ONNX Concat operator."""

  @classmethod
  def _prepare(cls, node: onnx_node.OnnxNode):
    super().prepare_attrs_dict(node, onnx_concat)

  @classmethod
  def version_13(cls, node: onnx_node.OnnxNode) -> Callable[..., Any]:
    """ONNX version_13 Concat op."""
    cls._prepare(node)
    return onnx_concat


@functools.partial(jit, static_argnames=('axis'))
def onnx_concat(*input_args, axis=0):
  """The internal jax impl for onnx Concat op."""
  return jnp.concatenate(input_args, axis=axis)
