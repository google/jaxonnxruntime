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
import functools
import inspect
from collections.abc import Callable, Sequence
from typing import Any

import onnx
from jax import jit
from jax import numpy as jnp
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node

import onnx

register_op = handler.register_op
Handler = handler.Handler
OnnxNode = onnx_node.OnnxNode


@handler.register_op("Cast")
class Cast(handler.Handler):
  """Implementation of the ONNX Cast operator."""

  @classmethod
  def _prepare(cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any):
    sig = inspect.signature(onnx_jax_impl)
    kwparams = [param.name for param in sig.parameters.values() if param.kind == inspect.Parameter.KEYWORD_ONLY]
    for name in kwparams:
      node.attrs_dict[name] = node.attrs.get(name, None)
    if not node.attrs_dict['from_type']:
      from_type = node.context_graph.value_info_dict[
          node.inputs[0]
      ].type.tensor_type.elem_type
      node.attrs_dict['from_type'] = from_type

  @classmethod
  def version_13(cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]) -> Callable[..., Any]:
    """ONNX version_13 Cast op."""
    cls._prepare(node, inputs, onnx_cast)
    return onnx_cast

@functools.partial(jit, static_argnames=("to", "from_type"))
def onnx_cast(x, *, to, from_type=None):
  if from_type is onnx.TensorProto.STRING or to is onnx.TensorProto.STRING:
    raise NotImplementedError(
        "Cast JAX version do not support STRING type yet."
    )
  to_type = tensor_dtype_to_jnp_dtype(to)
  from_type = tensor_dtype_to_jnp_dtype(from_type) if from_type else x.dtype
  try:
    return x.view(from_type).astype(to_type)
  except Exception as e:
    raise ValueError(
        f"onnx_cast can not support from_type = {from_type}, to_type ="
        f" {to_type}"
    ) from e


def tensor_dtype_to_jnp_dtype(tensor_type):
  if tensor_type is onnx.TensorProto.BFLOAT16:
    return jnp.bfloat16
  return jnp.dtype(onnx.helper.mapping.TENSOR_TYPE_TO_NP_TYPE[tensor_type])
