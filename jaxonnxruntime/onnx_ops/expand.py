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
"""Define ONNX Expand operator."""
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


@handler.register_op("Expand")
class Expand(handler.Handler):
  """Implementation of the ONNX Expand operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    if config.jaxort_only_allow_initializers_as_static_args:
      if node.inputs[1] not in node.context_graph.initializer_dict:
        raise ValueError(
            f"{node.inputs[1]} is not constant but used as a static argument "
            "`shape` when `jax.jit` the `Expand` operator. "
            "The jitted function gives wrong results if its value changes."
        )
      node.attrs_dict["shape"] = tuple(
          node.context_graph.initializer_dict[node.inputs[1]].tolist()
      )
    else:
      node.attrs_dict["shape"] = tuple(inputs[1].tolist())

  @classmethod
  def version_8(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_8 Expand op."""
    cls._prepare(node, inputs, onnx_expand)
    return onnx_expand

  @classmethod
  def version_13(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_13 Expand op."""
    cls._prepare(node, inputs, onnx_expand)
    return onnx_expand


@functools.partial(jit, static_argnames="shape")
def onnx_expand(*input_args, shape):
  """https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#Expand for more details."""
  data = input_args[0]
  return data * jnp.ones(shape, dtype=data.dtype)
