"""Define ONNX Tan operator."""
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


@handler.register_op("Tan")
class Tan(handler.Handler):
  """Implementation of the ONNX Tan operator."""

  @classmethod
  def _prepare(cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any):
    onnx_ops_utils.update_node_attrs_dict(node, onnx_jax_impl)

  @classmethod
  def version_7(cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]) -> Callable[..., Any]:
    """ONNX version_7 Tan op."""
    cls._prepare(node, inputs, onnx_tan)
    return onnx_tan

  @classmethod
  def version_22(cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]) -> Callable[..., Any]:
    """ONNX version_22 Tan op."""
    cls._prepare(node, inputs, onnx_tan)
    return onnx_tan


@functools.partial(jax.jit, static_argnames=())
def onnx_tan(*input_args):
  """https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#Tan for more details."""
  assert len(input_args) == 1
  data = input_args[0]
  return jnp.tan(data)
