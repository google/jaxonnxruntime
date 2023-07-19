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

"""Define ONNX CastLike operator."""
# pylint: disable=unused-argument
# pylint: disable=g-explicit-length-test
from collections.abc import Callable, Sequence
import functools
from typing import Any

from jax import jit
from jaxonnxruntime import config
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node
from jaxonnxruntime.core import onnx_utils


@handler.register_op("CastLike")
class CastLike(handler.Handler):
  """Implementation of the ONNX CastLike operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    from_type = cls._get_type(node, node.inputs[0], inputs[0])
    node.attrs_dict["from_type"] = from_type

  @classmethod
  def _get_type(
      cls, node: onnx_node.OnnxNode, input_name: str, input_value: Any
  ):
    if node.context_graph.value_info_dict.get(input_name) is not None:
      tensor_proto = node.context_graph.value_info_dict.get(input_name)
      res_type = onnx_utils.tensor_dtype_to_jnp_dtype(
          tensor_proto.type.tensor_type.elem_type
      )
    elif config.jaxort_only_allow_initializers_as_static_args:
      if node.context_graph.initializer_dict.get(input_name) is not None:
        tensor_proto = node.context_graph.initializer_dict.get(input_name)
        res_type = onnx_utils.tensor_dtype_to_jnp_dtype(tensor_proto.data_type)
      else:
        raise ValueError(
            "`config.jaxort_only_allow_initializers_as_static_args = True but "
            f"{input_name} tensor is not constant. We can not use it"
            "a static argument of the `Cast` operator. "
        )
    else:
      res_type = input_value.dtype
    return res_type

  @classmethod
  def version_15(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_15 CastLike op."""
    cls._prepare(node, inputs, onnx_castlike)
    return onnx_castlike

  @classmethod
  def version_19(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_19 CastLike op."""
    cls._prepare(node, inputs, onnx_castlike)
    return onnx_castlike


@functools.partial(jit, static_argnames=("from_type",))
def onnx_castlike(*input_args, from_type):
  """https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#CastLike for more details."""
  assert len(input_args) == 2
  inp, target = input_args
  return inp.view(from_type).astype(target.dtype)
