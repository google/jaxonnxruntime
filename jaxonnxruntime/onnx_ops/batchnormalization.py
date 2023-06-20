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
"""Define ONNX BatchNormalization operator."""
# pylint: disable=unused-argument
# pylint: disable=g-explicit-length-test
from collections.abc import Callable, Sequence
import functools
from typing import Any

from jax import jit
from jax import numpy as jnp
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node


@handler.register_op('BatchNormalization')
class BatchNormalization(handler.Handler):
  """Implementation of the ONNX BatchNormalization operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    node.attrs_dict['epsilon'] = node.attrs.get('epsilon', 1e-5)
    node.attrs_dict['momentum'] = node.attrs.get('momentum', 0.9)
    node.attrs_dict['training_mode'] = node.attrs.get('training_mode', 0)

  @classmethod
  def version_7(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_7 BatchNormalization op."""
    cls._prepare(node, inputs, onnx_batchnormalization)
    return onnx_batchnormalization

  @classmethod
  def version_9(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_9 BatchNormalization op."""
    cls._prepare(node, inputs, onnx_batchnormalization)
    return onnx_batchnormalization

  @classmethod
  def version_15(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_15 BatchNormalization op."""
    cls._prepare(node, inputs, onnx_batchnormalization)
    return onnx_batchnormalization


@functools.partial(
    jit, static_argnames=('epsilon', 'momentum', 'training_mode')
)
def onnx_batchnormalization(
    *input_args, epsilon: float, momentum: float, training_mode: int
):
  """https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#BatchNormalization for more details."""
  x, scale, b, input_mean, input_var = input_args

  dims_x = len(x.shape)
  dim_ones = (1,) * (dims_x - 2)
  scale = scale.reshape(-1, *dim_ones)
  b = b.reshape(-1, *dim_ones)
  input_mean = input_mean.reshape(-1, *dim_ones)
  input_var = input_var.reshape(-1, *dim_ones)

  if training_mode == 0:
    return (x - input_mean) / jnp.sqrt(input_var + epsilon) * scale + b
  else:
    raise NotImplementedError(
        'BatchNormalization with training_mode was not implemented yet.'
    )
