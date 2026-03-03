"""Define ONNX Mean operator."""
# pylint: disable=unused-argument
# pylint: disable=g-explicit-length-test
import functools
from collections.abc import Callable, Sequence
from typing import Any

import jax
from jax import numpy as jnp
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node
from jaxonnxruntime.onnx_ops import onnx_ops_utils


@handler.register_op("Mean")
class Mean(handler.Handler):
  """Implementation of the ONNX Mean operator."""

  @classmethod
  def _prepare(cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any):
    onnx_ops_utils.update_node_attrs_dict(node, onnx_jax_impl)

  @classmethod
  def version_1(cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]) -> Callable[..., Any]:
    """ONNX version_1 Mean op."""
    cls._prepare(node, inputs, onnx_mean)
    return onnx_mean

  @classmethod
  def version_6(cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]) -> Callable[..., Any]:
    """ONNX version_6 Mean op."""
    cls._prepare(node, inputs, onnx_mean)
    return onnx_mean

  @classmethod
  def version_8(cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]) -> Callable[..., Any]:
    """ONNX version_8 Mean op."""
    cls._prepare(node, inputs, onnx_mean)
    return onnx_mean

  @classmethod
  def version_13(cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]) -> Callable[..., Any]:
    """ONNX version_13 Mean op."""
    cls._prepare(node, inputs, onnx_mean)
    return onnx_mean


@functools.partial(jax.jit, static_argnames=())
def onnx_mean(*input_args):
  """Element-wise mean of input tensors."""
  # Stack all inputs and compute mean along the first axis
  # This computes (A + B + C + ...) / N
  stacked = jnp.stack(input_args, axis=0)
  return jnp.mean(stacked, axis=0)
