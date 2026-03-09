"""Define ONNX Sign operator."""
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


@handler.register_op("Sign")
class Sign(handler.Handler):
  """Implementation of the ONNX Sign operator."""

  @classmethod
  def _prepare(cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any):
    onnx_ops_utils.update_node_attrs_dict(node, onnx_jax_impl)

  @classmethod
  def version_9(cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]) -> Callable[..., Any]:
    """ONNX version_9 Sign op."""
    cls._prepare(node, inputs, onnx_sign)
    return onnx_sign

  @classmethod
  def version_13(cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]) -> Callable[..., Any]:
    """ONNX version_13 Sign op."""
    cls._prepare(node, inputs, onnx_sign)
    return onnx_sign


@functools.partial(jax.jit, static_argnames=())
def onnx_sign(*input_args):
  """https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#Sign for more details."""
  assert len(input_args) == 1
  data = input_args[0]
  return jnp.sign(data)
