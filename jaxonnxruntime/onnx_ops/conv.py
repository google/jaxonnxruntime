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
import copy
from collections.abc import Callable
import functools
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
  def version_11(cls, node: onnx_node.OnnxNode) -> Callable[..., Any]:
    """ONNX version_11 CONV op."""
    cls._prepare(node)
    return onnx_conv

  @classmethod
  def _prepare(cls, node: onnx_node.OnnxNode) -> None:
    super().prepare_attrs_dict(node, onnx_conv)
    if not node.attrs_dict['group']:
      node.attrs_dict["group"] = 1
    if "pads" in node.attrs:
      pads = node.attrs["pads"]
      # ONNX follows [x1_begin, x2_begin...x1_end, x2_end,...].
      # lax conv is a sequence of n (low, high) integer pairs.
      n = len(pads) // 2
      pads_new = ((pads[i], pads[i + n]) for i in range(n))
      node.attrs_dict["pads"] = tuple(pads_new)
    else:
      onnx_to_jax_pad_type = {
          "SAME_UPPER": "SAME",
          "VALID": "VALID",
          "SAME_LOWER": "SAME_LOWER",
      }
      if node.attrs["auto_pad"] not in onnx_to_jax_pad_type:
        raise ValueError(
            "Invalid auto_pad attribute: {}".format(node.attrs_dict["auto_pad"])
        )
      node.attrs_dict["pads"] = onnx_to_jax_pad_type[node.attrs["auto_pad"]]


@functools.partial(
    jit,
    static_argnames=("group", "kernel_shape", "pads", "strides", "dilations"),
)
def onnx_conv(
    *inputs,
    group: int = 1,
    kernel_shape: Optional[tuple[int, ...]] = None,
    pads: Any = "VALID",
    strides: Optional[tuple[int, ...]] = None,
    dilations: Optional[tuple[int, ...]] = None,
) -> jnp.ndarray:
  """JAX common impl of onnx Conv.

  Args:
    inputs: all those inputs. it include
      x: The input tensor.
      w: The weight tensor.
      b (optional): The bias tensor.
    group: The number of groups.
    kernel_shape: The kernel shape.
    pads: The padding.
    strides: The strides.
    dilations: The dilations.

  Returns:
    jax.numpy.ndarray: The output tensor.
  """
  assert len(inputs) == 2 or len(inputs) == 3
  if len(inputs) == 2:
    x, w = inputs
    b = None
  else:
    x, w, b = inputs
  kernel_shape = kernel_shape or w.shape
  spatial_size = w.ndim - 2
  strides = strides or tuple([1] * spatial_size)

  if b is not None:
    b = b.reshape([1, w.shape[0]] + [1] * spatial_size)
  else:
    b = jnp.array(0)

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
