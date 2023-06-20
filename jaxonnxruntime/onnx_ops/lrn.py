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
"""Define ONNX LRN operator."""
# pylint: disable=unused-argument
# pylint: disable=g-explicit-length-test
from collections.abc import Callable, Sequence
import functools
from typing import Any

from jax import jit
from jax import lax
from jax import numpy as jnp
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node


@handler.register_op("LRN")
class LRN(handler.Handler):
  """Implementation of the ONNX LRN operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    node.attrs_dict["alpha"] = node.attrs.get("alpha", float(0.0001))
    node.attrs_dict["beta"] = node.attrs.get("beta", float(0.75))
    node.attrs_dict["bias"] = node.attrs.get("bias", float(1.0))
    node.attrs_dict["size"] = int(node.attrs.get("size"))
    assert node.attrs_dict["size"] is not None

  @classmethod
  def version_1(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_1 LRN op."""
    cls._prepare(node, inputs, onnx_lrn)
    return onnx_lrn

  @classmethod
  def version_13(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_13 LRN op."""
    cls._prepare(node, inputs, onnx_lrn)
    return onnx_lrn


@functools.partial(jit, static_argnames=("alpha", "beta", "bias", "size"))
def onnx_lrn(*input_args, alpha, beta, bias, size):
  """https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#LRN for more details."""
  assert len(input_args) == 1
  x = input_args[0]
  shape = x.shape
  div = jnp.reshape(jnp.multiply(x, x), (shape[0], 1, shape[1], shape[2], -1))
  kernel_shape = (1, 1, size, 1, 1)
  strides = (1, 1, 1, 1, 1)
  pads = "SAME"
  div = (
      lax.reduce_window(
          div,
          jnp.array(0, dtype=div.dtype),
          lax.add,
          kernel_shape,
          strides,
          pads,
      )
      / size
  )
  div = jnp.reshape(div, shape)
  div = jnp.power(jnp.add(jnp.multiply(div, alpha), bias), beta)
  return jnp.true_divide(x, div)
