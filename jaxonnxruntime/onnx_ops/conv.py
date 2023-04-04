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

"""Define ONNX Conv operator."""
from collections.abc import Callable
import functools
import inspect
from typing import Any, Optional

from jax import jit
from jax import lax
import jax.numpy as jnp

from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node


@handler.register_op("Conv")
class Conv(handler.Handler):
  """Implementation of the ONNX Conv operator."""

  @classmethod
  def version_1(cls, node: onnx_node.OnnxNode) -> Callable[..., Any]:
    """ONNX version_1 CONV op."""
    cls._rewrite(node)
    cls._prepare(node)
    return onnx_conv

  @classmethod
  def version_11(cls, node: onnx_node.OnnxNode) -> Callable[..., Any]:
    """ONNX version_11 CONV op."""
    cls._rewrite(node)
    cls._prepare(node)
    return onnx_conv

  @classmethod
  def _rewrite(cls: Any, node: onnx_node.OnnxNode) -> None:
    """Rewrite the OnnxNode class for Jax implementation."""
    if "group" not in node.attrs:
      node.attrs["group"] = 1
    if "pads" in node.attrs:
      pads = node.attrs["pads"]
      # ONNX follows [x1_begin, x2_begin...x1_end, x2_end,...].
      # lax conv is a sequence of n (low, high) integer pairs.
      n = len(pads) // 2
      pads_new = ((pads[i], pads[i + n]) for i in range(n))
      node.attrs["pads"] = tuple(pads_new)
      if "auto_pads" in node.attrs:
        del node.attrs["auto_pads"]
    else:
      onnx_to_jax_pad_type = {
          "SAME_UPPER": "SAME",
          "VALID": "VALID",
          "SAME_LOWER": "SAME_LOWER",
      }
      if node.attrs["auto_pad"] not in onnx_to_jax_pad_type:
        raise ValueError(
            "Invalid auto_pad attribute: {}".format(node.attrs["auto_pad"])
        )
      node.attrs["pads"] = onnx_to_jax_pad_type[node.attrs["auto_pad"]]

  @classmethod
  def _prepare(cls, node: onnx_node.OnnxNode) -> None:
    args = list(inspect.signature(onnx_conv).parameters.keys())
    attrs = [node.attrs.get(k, None) for k in args[node.len_inputs :]]
    node.attrs_list.extend(attrs)


@functools.partial(
    jit,
    static_argnames=("group", "kernel_shape", "pads", "strides", "dilations"),
)
def onnx_conv(
    x: jnp.ndarray,
    w: jnp.ndarray,
    b: Optional[jnp.ndarray],
    group: Optional[int] = 1,
    kernel_shape: Optional[tuple[int, ...]] = None,
    pads: Any = "VALID",
    strides: Optional[tuple[int, ...]] = None,
    dilations: Optional[tuple[int, ...]] = None,
) -> jnp.ndarray:
  """JAX common impl of onnx Conv.

  Args:
    x (jax.numpy.ndarray): The input tensor.
    w (jax.numpy.ndarray): The weight tensor.
    b (jax.numpy.ndarray): The bias tensor.
    group (int): The number of groups.
    kernel_shape (tuple): The kernel shape.
    pads (tuple): The padding.
    strides (tuple): The strides.
    dilations (tuple): The dilations.

  Returns:
    jax.numpy.ndarray: The output tensor.
  """

  kernel_shape = kernel_shape or w.shape
  spatial_size = w.ndim - 2
  strides = strides or [1] * spatial_size

  if b is not None:
    b = b.reshape([1, w.shape[0]] + [1] * spatial_size)
  else:
    b = 0

  out = lax.conv_general_dilated(
      lhs=x,
      rhs=w,
      window_strides=strides,
      padding=pads,
      lhs_dilation=None,
      rhs_dilation=dilations,
      dimension_numbers=None,
      feature_group_count=group,
      batch_group_count=1,
  )
  return out + b
