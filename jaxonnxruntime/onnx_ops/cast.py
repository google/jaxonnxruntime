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
"""Define ONNX Cast operator."""
import logging
from functools import partial

from jax import jit
from jax import numpy as jnp
import onnx
from onnx import TensorProto


from jaxonnxruntime.core import handler, onnx_node

register_op = handler.register_op
Handler = handler.Handler
OnnxNode = onnx_node.OnnxNode


@register_op("Cast")
class Cast(Handler):
  """Implementation of the ONNX Cast operator."""

  @classmethod
  def version_1(cls, node: OnnxNode):
    cls._prepare(node)
    return onnx_cast

  @classmethod
  def version_6(cls, node: OnnxNode):
    cls._prepare(node)
    return onnx_cast

  @classmethod
  def version_9(cls, node: OnnxNode):
    cls._prepare(node)
    return onnx_cast

  @classmethod
  def version_13(cls, node: OnnxNode):
    cls._prepare(node)
    return onnx_cast

  @classmethod
  def _prepare(cls, node: OnnxNode):
    super().prepare_attrs_list(node, onnx_cast)
    if not node.attrs_list[-1]:
      from_type = node.context_graph.value_info_dict[
          node.inputs[0]].type.tensor_type.elem_type
      node.attrs_list[-1] = from_type


#@partial(jit, static_argnames=('to', 'from_type'))
def onnx_cast(x, to, from_type=None):
  if from_type is TensorProto.STRING or to is TensorProto.STRING:
    raise NotImplementedError(
        "Cast JAX version do not support STRING type yet.")
  to_type = tensor_dtype_to_jnp_dtype(to)
  from_type = tensor_dtype_to_jnp_dtype(from_type) if from_type else x.dtype
  try:
    return x.view(from_type).astype(to_type)
  except Exception as e:
    raise ValueError(
        f"onnx_cast can not support from_type = {from_type}, to_type = {to_type}"
    ) from e

def tensor_dtype_to_jnp_dtype(tensor_type):
  if tensor_type is TensorProto.BFLOAT16:
    return jnp.bfloat16
  return jnp.dtype(onnx.helper.tensor_dtype_to_np_dtype(tensor_type))
