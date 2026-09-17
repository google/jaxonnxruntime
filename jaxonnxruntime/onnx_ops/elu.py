"""Define ONNX Elu operator."""
# pylint: disable=unused-argument
# pylint: disable=g-explicit-length-test
import functools
from collections.abc import Callable, Sequence
from typing import Any

import jax
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node
from jaxonnxruntime.onnx_ops import onnx_ops_utils


@handler.register_op("Elu")
class Elu(handler.Handler):
  """Implementation of the ONNX Elu operator."""

  @classmethod
  def _prepare(cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any):
    node.attrs_dict['alpha'] = node.attrs.get(
    'alpha', 1.0
    )

  @classmethod
  def version_1(cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]) -> Callable[..., Any]:
    """ONNX version_1 Elu op."""
    cls._prepare(node, inputs, onnx_elu)
    return onnx_elu

  @classmethod
  def version_6(cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]) -> Callable[..., Any]:
    """ONNX version_6 Elu op."""
    cls._prepare(node, inputs, onnx_elu)
    return onnx_elu

  @classmethod
  def version_22(cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]) -> Callable[..., Any]:
    """ONNX version_22 Elu op."""
    cls._prepare(node, inputs, onnx_elu)
    return onnx_elu


@functools.partial(jax.jit, static_argnames=())
def onnx_elu(*input_args, alpha):
  """https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#Elu for more details."""
  assert len(input_args) == 1
  data = input_args[0]
  return jax.nn.elu(data, alpha)
