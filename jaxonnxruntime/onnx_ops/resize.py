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

"""Define ONNX Resize operator."""
from collections.abc import Callable, Sequence
import functools
from typing import Any
import jax
from jax import numpy as jnp
from jaxonnxruntime.core import config_class

config = config_class.config
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node
import numpy as np


@handler.register_op('Resize')
class Resize(handler.Handler):
  """Implementation of the ONNX Resize operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    # Extract attributes
    # attrs are already strings in this jaxonnxruntime version
    mode = node.attrs.get('mode', 'nearest')
    if isinstance(mode, bytes):
        mode = mode.decode('utf-8')
    node.attrs_dict['mode'] = mode
    
    coordinate_transformation_mode = node.attrs.get('coordinate_transformation_mode', 'half_pixel')
    if isinstance(coordinate_transformation_mode, bytes):
        coordinate_transformation_mode = coordinate_transformation_mode.decode('utf-8')
    node.attrs_dict['coordinate_transformation_mode'] = coordinate_transformation_mode
    
    # Resize has inputs: X, roi, scales, sizes
    # We need to determine target 'sizes'
    if len(inputs) >= 4 and inputs[3] is not None:
        node.attrs_dict['sizes'] = tuple(inputs[3].astype(int).tolist())
    elif len(inputs) >= 3 and inputs[2] is not None:
        scales = inputs[2]
        x_shape = inputs[0].shape
        sizes = [int(x_shape[i] * scales[i]) for i in range(len(x_shape))]
        node.attrs_dict['sizes'] = tuple(sizes)
    else:
        # Fallback: check if sizes/scales are in constant dict
        constant_dict = node.context_graph.get_constant_dict()
        if len(node.inputs) >= 4 and node.inputs[3] in constant_dict:
             node.attrs_dict['sizes'] = tuple(constant_dict[node.inputs[3]].astype(int).tolist())
        elif len(node.inputs) >= 3 and node.inputs[2] in constant_dict:
             scales = constant_dict[node.inputs[2]]
             x_shape = inputs[0].shape
             sizes = [int(x_shape[i] * scales[i]) for i in range(len(x_shape))]
             node.attrs_dict['sizes'] = tuple(sizes)
        else:
             # If still not found, we might have to wait for runtime shape or it's a dynamic resize.
             # However, for JAX, we prefer static sizes.
             raise ValueError(f"Resize node {node.name} needs valid 'scales' or 'sizes'.")

  @classmethod
  def version_10(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_10 Resize op."""
    cls._prepare(node, inputs, onnx_resize)
    return onnx_resize

  @classmethod
  def version_11(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_11 Resize op."""
    cls._prepare(node, inputs, onnx_resize)
    return onnx_resize

  @classmethod
  def version_13(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_13 Resize op."""
    cls._prepare(node, inputs, onnx_resize)
    return onnx_resize

  @classmethod
  def version_18(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_18 Resize op."""
    cls._prepare(node, inputs, onnx_resize)
    return onnx_resize


@functools.partial(jax.jit, static_argnames=('sizes', 'mode', 'coordinate_transformation_mode'))
def onnx_resize(*input_args, sizes, mode, coordinate_transformation_mode):
  """The impl for Resize."""
  x = input_args[0]
  
  # Map ONNX modes to JAX modes
  # ONNX modes: 'nearest', 'linear', 'cubic'
  # JAX modes: 'nearest', 'linear', 'bilinear', 'trilinear', 'cubic', 'lanczos3', 'lanczos5'
  jax_method = mode
  if jax_method == 'linear':
      # If rank is 4 (N, C, H, W), 'bilinear' is more appropriate for 2D spatial resize
      if x.ndim == 4:
          jax_method = 'bilinear'
      else:
          jax_method = 'linear'
  
  return jax.image.resize(x, shape=sizes, method=jax_method)
