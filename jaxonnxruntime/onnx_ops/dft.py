# Copyright 2026 The Jaxonnxruntime Authors.
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

"""Define ONNX DFT operator."""

# pylint: disable=unused-argument
# pylint: disable=g-explicit-length-test
from collections.abc import Callable, Sequence
import functools
from typing import Any

import jax
from jax import numpy as jnp
from jaxonnxruntime.core import config_class
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node

config = config_class.config


@handler.register_op('DFT')
class DFT(handler.Handler):
  """Implementation of the ONNX DFT operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], version: int
  ):
    node.attrs_dict['onesided'] = node.attrs.get('onesided', 0)
    node.attrs_dict['inverse'] = node.attrs.get('inverse', 0)

    # Handle dft_length (input 1)
    dft_length = None
    if len(inputs) > 1 and inputs[1] is not None:
      if config.jaxort_only_allow_initializers_as_static_args:
        if node.inputs[1] not in node.context_graph.get_constant_dict():
          raise ValueError(f'{node.inputs[1]} must be constant')
        dft_length = int(
            node.context_graph.get_constant_dict()[node.inputs[1]].item()
        )
      else:
        dft_length = int(inputs[1].item())
    node.attrs_dict['dft_length'] = dft_length

    # Handle axis
    if version == 17:
      node.attrs_dict['axis'] = node.attrs.get('axis', 1)
    elif version == 20:
      axis = -2  # default
      if len(inputs) > 2 and inputs[2] is not None:
        if config.jaxort_only_allow_initializers_as_static_args:
          if node.inputs[2] not in node.context_graph.get_constant_dict():
            raise ValueError(f'{node.inputs[2]} must be constant')
          axis = int(
              node.context_graph.get_constant_dict()[node.inputs[2]].item()
          )
        else:
          axis = int(inputs[2].item())
      node.attrs_dict['axis'] = axis

  @classmethod
  def version_17(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_17 DFT op."""
    cls._prepare(node, inputs, version=17)
    return onnx_dft

  @classmethod
  def version_20(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_20 DFT op."""
    cls._prepare(node, inputs, version=20)
    return onnx_dft


@functools.partial(
    jax.jit, static_argnames=('dft_length', 'axis', 'onesided', 'inverse')
)
def onnx_dft(*input_args, dft_length, axis, onesided, inverse):
  x = input_args[0]

  # Normalize axis
  axis = axis % x.ndim

  if x.shape[-1] == 1:
    signal = x[..., 0]
  elif x.shape[-1] == 2:
    signal = x[..., 0] + 1j * x[..., 1]
  else:
    raise ValueError('Last dimension must be 1 or 2')

  if dft_length is None:
    dft_length = x.shape[axis]

  if inverse:
    result = jnp.fft.ifft(signal, n=dft_length, axis=axis)
  else:
    result = jnp.fft.fft(signal, n=dft_length, axis=axis)

  res_real = jnp.real(result)
  res_imag = jnp.imag(result)
  res = jnp.stack([res_real, res_imag], axis=-1)

  if onesided:
    slices = [slice(None)] * res.ndim
    slices[axis] = slice(0, dft_length // 2 + 1)
    res = res[tuple(slices)]

  return res
