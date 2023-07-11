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
"""Define ONNX Range operator."""
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


@handler.register_op("Range")
class Range(handler.Handler):
  """Implementation of the ONNX Range operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    effective_inputs = []
    if config.jaxort_only_allow_initializers_as_static_args:
      for inp in node.inputs[:]:
        if inp not in node.context_graph.initializer_dict:
          raise ValueError(
              f"{inp} is not constant but used as a static argument "
              "when `jax.jit` the `Range` operator. "
              "The jitted function gives wrong results if its value changes."
          )
        effective_inputs.append(node.context_graph.initializer_dict[inp])
    else:
      effective_inputs = inputs
    node.attrs_dict["start"] = effective_inputs[0].item()
    node.attrs_dict["limit"] = effective_inputs[1].item()
    node.attrs_dict["delta"] = effective_inputs[2].item()
    node.attrs_dict["dtype"] = effective_inputs[0].dtype

  @classmethod
  def version_11(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_11 Range op."""
    cls._prepare(node, inputs, onnx_range)
    return onnx_range


@functools.partial(jit, static_argnames=("start", "limit", "delta", "dtype"))
def onnx_range(*_, start, limit, delta, dtype):
  """https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#Range for more details."""
  return jnp.arange(start, stop=limit, step=delta, dtype=dtype)
