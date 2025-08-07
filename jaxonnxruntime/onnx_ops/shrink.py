"""Define ONNX Shrink operator."""
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


@handler.register_op("Shrink")
class Shrink(handler.Handler):
  """Implementation of the ONNX Shrink operator."""

  @classmethod
  def _prepare(cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any):
    node.attrs_dict['bias'] = node.attrs.get('bias', 0.0)
    node.attrs_dict['lambd'] = node.attrs.get('lambd', 0.0)

  @classmethod
  def version_9(cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]) -> Callable[..., Any]:
    """ONNX version_9 Shrink op."""
    cls._prepare(node, inputs, onnx_shrink)
    return onnx_shrink


@functools.partial(jax.jit, static_argnames=())
def onnx_shrink(*input_args, bias, lambd):
  """https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#Shrink for more details."""
  assert len(input_args) == 1
  data = input_args[0]
  return jnp.where(
      data < -lambd,
      data + bias,
      jnp.where(data > lambd, data - bias, 0),
      ).astype(data.dtype)
