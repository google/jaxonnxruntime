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
"""Define ONNX ReduceSum operator."""
# pylint: disable=g-explicit-length-test
from collections.abc import Callable, Sequence
import functools
from typing import Any
import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node


@handler.register_op('ReduceSum')
class ReduceSum(handler.Handler):
  """Implementation of the ONNX ReduceSum operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    node.attrs_dict['axes'] = node.attrs.get('axes')
    if len(inputs) >= 2:
      node.attrs_dict['axes'] = tuple(inputs[1].tolist())
      node.attrs_dict['axes'] = (
          None if len(node.attrs_dict['axes']) == 0 else node.attrs_dict['axes']
      )
    node.attrs_dict['keepdims'] = node.attrs.get('keepdims', 1)
    node.attrs_dict['noop_with_empty_axes'] = node.attrs.get(
        'noop_with_empty_axes', 0
    )

  @classmethod
  def version_1(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_1 ReduceSum op."""
    cls._prepare(node, inputs, onnx_reducesum)
    return onnx_reducesum

  @classmethod
  def version_13(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_13 ReduceSum op."""
    cls._prepare(node, inputs, onnx_reducesum)
    return onnx_reducesum

  @classmethod
  def pallas_version_1(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_1 ReduceSum op with Pallas."""
    cls._prepare(node, inputs, onnx_reducesum_pallas)
    return onnx_reducesum_pallas

  @classmethod
  def pallas_version_13(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_13 ReduceSum op with Pallas."""
    cls._prepare(node, inputs, onnx_reducesum_pallas)
    return onnx_reducesum_pallas


@functools.partial(
    jax.jit, static_argnames=('axes', 'keepdims', 'noop_with_empty_axes')
)
def onnx_reducesum(
    *input_args,
    axes=None,
    keepdims=1,
    noop_with_empty_axes=0,
):
  """The impl for https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#ReduceSum."""
  assert len(input_args) == 1 or len(input_args) == 2
  data = input_args[0]
  if axes is None and noop_with_empty_axes > 0:
    return data
  return jnp.sum(data, axis=axes, keepdims=keepdims > 0)


@functools.partial(
    jax.jit, static_argnames=('axes', 'keepdims', 'noop_with_empty_axes')
)
def onnx_reducesum_pallas(
    *input_args,
    axes=None,
    keepdims=1,
    noop_with_empty_axes=0,
):
  """The internal jax impl for onnx ReduceSum op using Pallas."""
  assert len(input_args) == 1 or len(input_args) == 2
  data = input_args[0]

  if axes is None and noop_with_empty_axes > 0:
    return data

  # Normalize axes handled by _prepare and keyword args.

  # Fallback conditions
  # 1. Only support 2D input for now
  if len(data.shape) != 2:
    return jnp.sum(data, axis=axes, keepdims=keepdims > 0)

  # 2. Only support reducing over the last axis (axis=1 or axis=-1)
  if axes != (1,) and axes != (-1,):
    return jnp.sum(data, axis=axes, keepdims=keepdims > 0)

  # 3. Fallback for unsupported dtypes
  supported_dtypes = [
      jnp.float32,
      jnp.bfloat16,
      jnp.int16,
      jnp.uint16,
      jnp.int32,
      jnp.uint32,
  ]
  if data.dtype not in supported_dtypes:
    return jnp.sum(data, axis=axes, keepdims=keepdims > 0)

  # 4. Handle noop_with_empty_axes
  if not axes and noop_with_empty_axes > 0:
    return data

  M, N = data.shape
  keepdims_bool = keepdims > 0

  def reduce_sum_kernel(x_ref, z_ref):
    z_ref[...] = jnp.sum(x_ref[...], axis=1, keepdims=keepdims_bool)

  out_shape = jnp.sum(data, axis=axes, keepdims=keepdims_bool).shape

  in_specs = [pl.BlockSpec((8, N), lambda i: (i, 0))]
  out_specs = pl.BlockSpec(
      (8, 1) if keepdims_bool else (8,),
      lambda i: (i, 0) if keepdims_bool else (i,),
  )

  return pl.pallas_call(
      reduce_sum_kernel,
      out_shape=jax.ShapeDtypeStruct(out_shape, data.dtype),
      grid=(M // 8,),
      in_specs=in_specs,
      out_specs=out_specs,
  )(data)
