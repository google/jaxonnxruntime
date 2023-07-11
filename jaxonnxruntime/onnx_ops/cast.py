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

from collections.abc import Callable, Sequence
import functools
from typing import Any, Optional

import jax
from jax import numpy as jnp
from jaxonnxruntime import config
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node
from jaxonnxruntime.core import onnx_utils

import onnx


register_op = handler.register_op
Handler = handler.Handler
OnnxNode = onnx_node.OnnxNode


@handler.register_op("Cast")
class Cast(handler.Handler):
  """Implementation of the ONNX Cast operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    node.attrs_dict["to"] = node.attrs.get("to", int)
    if node.context_graph.value_info_dict.get(node.inputs[0]) is not None:
      tensor_proto = node.context_graph.value_info_dict.get(node.inputs[0])
      from_type = onnx_utils.tensor_dtype_to_jnp_dtype(
          tensor_proto.type.tensor_type.elem_type
      )
    elif config.jaxort_only_allow_initializers_as_static_args:
      if node.context_graph.initializer_dict.get(node.inputs[0]) is not None:
        tensor_proto = node.context_graph.initializer_dict.get(node.inputs[0])
        from_type = onnx_utils.tensor_dtype_to_jnp_dtype(tensor_proto.data_type)
      else:
        raise ValueError(
            "`config.jaxort_only_allow_initializers_as_static_args = True but "
            f"{node.inputs[0]} tensor is not constant. We can not use it"
            "a static argument of the `Cast` operator. "
        )
    else:
      from_type = inputs[0].dtype
    node.attrs_dict["from_type"] = from_type

  @classmethod
  def version_9(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_9 Cast op."""
    cls._prepare(node, inputs, onnx_cast)
    return onnx_cast

  @classmethod
  def version_13(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_13 Cast op."""
    cls._prepare(node, inputs, onnx_cast)
    return onnx_cast

  @classmethod
  def version_19(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_19 Cast op."""
    cls._prepare(node, inputs, onnx_cast)
    return onnx_cast


@functools.partial(jax.jit, static_argnames=("to", "from_type"))
def onnx_cast(
    x: jax.Array,
    *,
    to: onnx.TensorProto.DataType,
    from_type: Optional[jnp.dtype],
) -> jax.Array:
  """https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#Cast for more details."""
  if from_type is onnx.TensorProto.STRING or to is onnx.TensorProto.STRING:
    raise NotImplementedError(
        "Cast JAX version do not support STRING type yet."
    )
  to_type = onnx_utils.tensor_dtype_to_jnp_dtype(to)
  try:
    return x.view(from_type).astype(to_type)
  except Exception as e:
    raise ValueError(
        f"onnx_cast cannot support from_type = {from_type}, to_type ="
        f" {to_type}"
    ) from e
