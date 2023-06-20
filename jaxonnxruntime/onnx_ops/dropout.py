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
"""Define ONNX Dropout operator."""
# pylint: disable=unused-argument
# pylint: disable=g-explicit-length-test
from collections.abc import Callable, Sequence
import functools
from typing import Any

from jax import jit
from jax import lax
from jax import numpy as jnp
from jax import random
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node


@handler.register_op("Dropout")
class Dropout(handler.Handler):
  """Implementation of the ONNX Dropout operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    node.attrs_dict["seed"] = node.attrs.get(
        "seed", 0
    )  # TODO(lijinning): need to randomly generate a seed
    node.attrs_dict["require_mask"] = True if len(node.outputs) > 1 else False
    node.attrs_dict["ratio"] = 0.5 if len(inputs) == 1 else inputs[1].item()
    node.attrs_dict["training_mode"] = False if len(inputs) < 3 else inputs[2]

  @classmethod
  def version_7(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_7 Dropout op."""
    cls._prepare(node, inputs, onnx_dropout)
    return onnx_dropout

  @classmethod
  def version_10(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_10 Dropout op."""
    cls._prepare(node, inputs, onnx_dropout)
    return onnx_dropout

  @classmethod
  def version_12(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_12 Dropout op."""
    cls._prepare(node, inputs, onnx_dropout)
    return onnx_dropout

  @classmethod
  def version_13(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_13 Dropout op."""
    cls._prepare(node, inputs, onnx_dropout)
    return onnx_dropout


@functools.partial(
    jit, static_argnames=("ratio", "training_mode", "require_mask")
)
def onnx_dropout(*input_args, ratio, training_mode, seed, require_mask):
  """https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#Dropout for more details."""
  data = input_args[0]
  deterministic = not training_mode

  if (ratio == 0.0) or deterministic:
    if require_mask:
      return data, jnp.ones_like(data).astype(bool)
    else:
      return data

  # Prevent gradient NaNs in 1.0 edge-case.
  if ratio == 1.0:
    if require_mask:
      return jnp.zeros_like(data), jnp.zeros_like(data).astype(bool)
    else:
      return jnp.zeros_like(data)

  keep_prob = 1.0 - ratio
  broadcast_shape = list(data.shape)
  rng = random.PRNGKey(seed)
  mask = random.bernoulli(rng, p=keep_prob, shape=broadcast_shape)
  mask = jnp.broadcast_to(mask, data.shape)
  if require_mask:
    return lax.select(
        mask, data / keep_prob, jnp.zeros_like(data)
    ), mask.astype(bool)
  else:
    return lax.select(mask, data / keep_prob, jnp.zeros_like(data))
