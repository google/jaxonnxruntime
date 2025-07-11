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

"""Define ONNX NonZero operator."""
from collections.abc import Callable, Sequence
import functools
import logging
from typing import Any

import jax
from jax import numpy as jnp
from jaxonnxruntime.core import config_class
from jaxonnxruntime.onnx_ops import onnx_ops_utils

config = config_class.config
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node


@handler.register_op("NonZero")
class NonZero(handler.Handler):
  """Implementation of the ONNX NonZero operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    onnx_ops_utils.update_node_attrs_dict(node, onnx_jax_impl)

    assert len(inputs) == 1
    if config.jaxort_nonzero_use_fully_padding:
      node.attrs_dict["size"] = inputs[0].size
    if node.attrs_dict["size"] is None:
      raise ValueError(
          "NonZero Jax implementation must have static size attribute but not."
      )

  @classmethod
  def version_9(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_9 NonZero op."""
    cls._prepare(node, inputs, onnx_nonzero)
    return onnx_nonzero

  @classmethod
  def version_13(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_13 NonZero op."""
    cls._prepare(node, inputs, onnx_nonzero)
    return onnx_nonzero


@functools.partial(jax.jit, static_argnames="size")
def onnx_nonzero(*input_args, size):
  """The impl for https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#NonZero."""
  assert len(input_args) == 1
  logging.warning("onnx_nonzero cannot support jax.jit mode.")
  (x,) = input_args
  return jnp.stack(jnp.nonzero(x, size=size))
