# Copyright 2026 The Jaxonnxruntime Authors.
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
"""Define ONNX Add operator."""
from collections.abc import Callable, Sequence
import functools
from typing import Any

import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node
from jaxonnxruntime.onnx_ops import onnx_ops_utils


@handler.register_op("Add")
class Add(handler.Handler):
  """Implementation of the ONNX Add operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    onnx_ops_utils.update_node_attrs_dict(node, onnx_jax_impl)

  @classmethod
  def version_1(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_1 Add op."""
    cls._prepare(node, inputs, onnx_add)
    return onnx_add

  @classmethod
  def version_6(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_6 Add op."""
    cls._prepare(node, inputs, onnx_add)
    return onnx_add

  @classmethod
  def version_7(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_7 Add op."""
    cls._prepare(node, inputs, onnx_add)
    return onnx_add

  @classmethod
  def version_13(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_13 Add op."""
    cls._prepare(node, inputs, onnx_add)
    return onnx_add

  @classmethod
  def version_14(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_14 Add op."""
    cls._prepare(node, inputs, onnx_add)
    return onnx_add

  @classmethod
  def pallas_version_1(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_1 Add op with Pallas."""
    cls._prepare(node, inputs, onnx_add_pallas)
    return onnx_add_pallas

  @classmethod
  def pallas_version_6(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_6 Add op with Pallas."""
    cls._prepare(node, inputs, onnx_add_pallas)
    return onnx_add_pallas

  @classmethod
  def pallas_version_7(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_7 Add op with Pallas."""
    cls._prepare(node, inputs, onnx_add_pallas)
    return onnx_add_pallas

  @classmethod
  def pallas_version_13(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_13 Add op with Pallas."""
    cls._prepare(node, inputs, onnx_add_pallas)
    return onnx_add_pallas

  @classmethod
  def pallas_version_14(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_14 Add op with Pallas."""
    cls._prepare(node, inputs, onnx_add_pallas)
    return onnx_add_pallas


@functools.partial(jax.jit, static_argnames=())
def onnx_add_pallas(*input_args):
  """The internal jax impl for onnx Add op using Pallas."""
  assert len(input_args) == 2
  a, b = input_args
  # Fallback to jnp.add if shapes don't match
  # (broadcasting not supported in Pallas yet)
  if a.shape != b.shape:
    return jnp.add(a, b)

  # Fallback to jnp.add for unsupported dtypes on TPU Pallas
  supported_dtypes = [
      jnp.float32,
      jnp.bfloat16,
      jnp.int16,
      jnp.uint16,
      jnp.int32,
      jnp.uint32,
  ]
  if a.dtype not in supported_dtypes:
    return jnp.add(a, b)

  def add_kernel(x_ref, y_ref, z_ref):
    z_ref[...] = x_ref[...] + y_ref[...]

  return pl.pallas_call(
      add_kernel, out_shape=jax.ShapeDtypeStruct(a.shape, a.dtype)
  )(a, b)


@functools.partial(jax.jit, static_argnames=())
def onnx_add(*input_args):
  """The internal jax impl for onnx Add op."""
  assert len(input_args) == 2
  a, b = input_args
  return jnp.add(a, b)
