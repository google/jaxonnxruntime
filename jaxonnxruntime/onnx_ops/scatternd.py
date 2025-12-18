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

"""Define ONNX ScatterND operator."""

# pylint: disable=unused-argument
# pylint: disable=g-explicit-length-test
from collections.abc import Callable, Sequence
import functools
from typing import Any

import jax
from jax import numpy as jnp
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node


@handler.register_op("ScatterND")
class ScatterND(handler.Handler):
  """Implementation of the ONNX ScatterND operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    reduction_map = {
        None: "set",
        "add": "add",
        "mul": "multiply",
        "max": "max",
        "min": "min",
    }
    node.attrs_dict["reduction"] = reduction_map[
        node.attrs.get("reduction", None)
    ]

  @classmethod
  def version_11(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_11 ScatterND op."""
    cls._prepare(node, inputs, onnx_scatternd)
    return onnx_scatternd

  @classmethod
  def version_13(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_13 ScatterND op."""
    cls._prepare(node, inputs, onnx_scatternd)
    return onnx_scatternd

  @classmethod
  def version_16(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_16 ScatterND op."""
    cls._prepare(node, inputs, onnx_scatternd)
    return onnx_scatternd

  @classmethod
  def version_18(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_18 ScatterND op."""
    cls._prepare(node, inputs, onnx_scatternd)
    return onnx_scatternd


@functools.partial(jax.jit, static_argnames="reduction")
def onnx_scatternd(*input_args, reduction: str):
  """Implements the ONNX ScatterND operator.

  Updates scattered elements of a data tensor according to indices and updates.
  The shapes of (data, indices, updates = input_args), and the return value are
  illustrated in the example below (q=3, r-k=3).

  Args:
    data: An r-dimensional tensor with k index dimension and (r-k) update
      dimensions.
              Example: [i1,i2,..,ik, x1,x2,x3] [-----k-----|--(r-k)--]
    indices: A q-dimensional tensor with j1*j2*..*j_{q-1} k-tuples, where each
      k-tuple is a partial index for the data.
              Example: [ j1,j2,  k ] [-(q-1)-|-1-]
    updates: A tensor with j1*j2*..*j_{q-1} partial updates for the data. The
      first (q-1) dimensions correspond to the first dimensions of indices. The
      last  (r-k) dimensions correspond to the last dimensions of data.
              Example: [ j1,j2, x1,x2,x3 ] [-(q-1)-|--(r-k)--]
    reduction: How to combine the updates with the data (e.g., set, add).

  Returns:
    A copy of data (i.e., same shape), but with the updates applied.
              Example  [i1,i2,..,ik, x1,x2,x3]

  https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#ScatterND for more
  details.
  """

  assert len(input_args) == 3
  data, indices, updates = input_args

  k = indices.shape[-1]
  r = len(data.shape)
  q = len(indices.shape)
  assert len(updates.shape) == q - 1 + r - k

  assert (
      indices.shape[: q - 1] == updates.shape[: q - 1]
  ), "first q-1 dims of indicies and updates must match"
  assert (
      data.shape[k:] == updates.shape[q - 1 :]
  ), "last dimensions of data and updates must match"
  assert r - k >= 0, f"expected non-negative but was: {r-k}"

  # flatten indices (j1, j2, .., j_{q-1}, k) -> (z, k)
  indices = jnp.reshape(indices, (-1, k))
  z = indices.shape[0]

  # (z, k) -> (k, z) -> (k, z, 1,..,1)
  indices = jnp.transpose(indices)
  assert indices.shape == (k, z)
  indices = jnp.expand_dims(indices, axis=list(range(2, 2 + r - k)))
  # (k, z, 1,..,1) -> [(z, 1,..,1), (z, 1,..,1), ..] with len k
  indices = list(indices)

  # reshape updates (j1, j2, .., j_{q-1}, x1, x2, ...) -> (z, x1, x2, ...)
  updates_shape = (z,) + updates.shape[q - 1 :]
  updates = jnp.reshape(updates, updates_shape)

  # e.g., for (r-k)=3 and z updates:
  # [(1,1,1,1), (1,range(x1),1,1), (1,1,range(x2),1), (1,1,1,range(x3))]
  #   L----------L------------------L------------------L----> dim for z
  idx = list(
      jnp.meshgrid(
          *(jnp.arange(n) for n in [1] + list(data.shape[k:])),
          sparse=True,
          indexing="ij",
      )
  )
  assert idx[0].ndim == (r - k) + 1

  # idx in `data.at[idx]` is a tuple of length data.ndim, where each element
  # is an array containing indices for a specific dimension of the data.
  # The first k dimensions contain the filters from `indices`,
  # the remaining dimensions select all entries.
  # [(z,1,1,1), (z,1,1,1)..] + [(1,range(x1),1,1), (1,1,range(x2),1), (1,1,1,range(x3))]
  # ------------k----------      ----------------------(r-k)---------------------------
  idx = indices + idx[1:]

  assert len(idx) == data.ndim, f"{len(idx)} != {data.ndim}"
  assert (
      len(set(i.ndim for i in idx)) == 1
  ), f"all idx must have the same ndim but were: {set(i.ndim for i in idx)}"

  select_shape = jnp.broadcast_shapes(*[i.shape for i in idx])
  assert select_shape == updates.shape, (
      "Shape of index-selected data must match the shape of the update:"
      f" {select_shape}!={updates.shape}."
  )

  out = getattr(data.at[tuple(idx)], reduction)(
      updates, indices_are_sorted=False
  )
  return out
