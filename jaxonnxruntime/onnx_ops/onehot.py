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
"""Define ONNX OneHot operator."""
# pylint: disable=unused-argument
# pylint: disable=g-explicit-length-test

from collections.abc import Callable, Sequence
import functools
from typing import Any

import jax
from jax import jit
from jax import numpy as jnp
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node


@handler.register_op("OneHot")
class OneHot(handler.Handler):
  """Implementation of the ONNX OneHot operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    node.attrs_dict["axis"] = node.attrs.get("axis", -1)
    node.attrs_dict["depth"] = int(inputs[1])

  @classmethod
  def version_11(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_11 OneHot op."""
    cls._prepare(node, inputs, onnx_onehot)
    return onnx_onehot


@functools.partial(jit, static_argnames=("depth", "axis"))
def onnx_onehot(*input_args, depth, axis):
  """https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#OneHot for more details."""
  assert len(input_args) == 3
  indices, _, values = input_args
  indices = jnp.mod(indices, depth)
  encode = jax.nn.one_hot(indices, depth, axis=axis)
  encode = encode * (values[1] - values[0]) + values[0]
  return encode.astype(values.dtype)
