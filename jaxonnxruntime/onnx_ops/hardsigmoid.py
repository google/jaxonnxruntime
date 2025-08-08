"""Define ONNX HardSigmoid operator."""
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


@handler.register_op("HardSigmoid")
class HardSigmoid(handler.Handler):
  """Implementation of the ONNX HardSigmoid operator."""

  @classmethod
  def _prepare(cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any):
    node.attrs_dict["alpha"] = node.attrs.get("alpha", 0.2)
    node.attrs_dict["beta"] = node.attrs.get("beta", 0.5)

  @classmethod
  def version_1(cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]) -> Callable[..., Any]:
    """ONNX version_1 HardSigmoid op."""
    cls._prepare(node, inputs, onnx_hardsigmoid)
    return onnx_hardsigmoid

  @classmethod
  def version_6(cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]) -> Callable[..., Any]:
    """ONNX version_6 HardSigmoid op."""
    cls._prepare(node, inputs, onnx_hardsigmoid)
    return onnx_hardsigmoid

  @classmethod
  def version_22(cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]) -> Callable[..., Any]:
    """ONNX version_22 HardSigmoid op."""
    cls._prepare(node, inputs, onnx_hardsigmoid)
    return onnx_hardsigmoid


@functools.partial(jax.jit, static_argnames=())
def onnx_hardsigmoid(*input_args, alpha, beta):
  """https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#HardSigmoid for more details."""
  assert len(input_args) == 1
  data = input_args[0]
  return jnp.maximum(0, jnp.minimum(1, data * alpha + beta)).astype(data.dtype)
