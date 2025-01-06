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
    node.attrs_dict["reduction"] = node.attrs.get("reduction", None)

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
              Example: [i1,i2,..,ik, x1,x2,x3]
                       [-----k-----|--(r-k)--]
    indices: A q-dimensional tensor with j1*j2*..*j_{q-1} k-tuples, where each
      k-tuple is a partial index for the data.
              Example: [ j1,j2,  k ]
                       [-(q-1)-|-1-]
    updates: A tensor with j1*j2*..*j_{q-1} partial updates for the data.
      The first (q-1) dimensions correspond to the first dimensions of indices.
      The last  (r-k) dimensions correspond to the last dimensions of data.
              Example: [ j1,j2, x1,x2,x3 ]
                       [-(q-1)-|--(r-k)--]
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

  assert (
      indices.shape[: q - 1] == updates.shape[: q - 1]
  ), "first dimensions of indicies and updates must match"
  assert (
      data.shape[k:] == updates.shape[q - 1 :]
  ), "last dimensions of data and updates must match"
  assert r - k >= 0, f"expected non-negative but was: {r-k}"
  assert len(updates.shape) == q - 1 + r - k

  # flatten z indices (z1, z2, ..., zn, k) -> (z, k)
  indices = jnp.reshape(indices, (-1, k))
  z = indices.shape[0]

  # (z, k) -> (k, z) -> (k, z, 1, ..., 1)
  indices = jnp.transpose(indices)
  assert indices.shape == (k, z)
  indices = jnp.expand_dims(indices, axis=list(range(2, 2 + r - k)))

  # (k, z, 1, ..., 1) -> [(z, 1, ..., 1), (z, 1, ..., 1), ...] with len k
  indices = list(indices)

  # reshape updates (z1, z2, ..., zn, x1, x2, ...) -> (z, x1, x2, ...)
  updates_shape = (z,) + updates.shape[q - 1 :]
  updates = jnp.reshape(updates, updates_shape)

  # e.g., [(range(n0),1,1), (1,range(n1),1), (1,1,range(n2))] for 3d data
  idx = jnp.meshgrid(
      *(jnp.arange(n) for n in data.shape), sparse=True, indexing="ij"
  )

  # in first k dimensions, instead of selecting all elems with (range(n0),1,1),
  # we only select the ones specified by indices
  for i in range(k):
    idx[i] = indices[i]

  out = getattr(data.at[tuple(idx)], reduction or "set")(updates)

  return out
