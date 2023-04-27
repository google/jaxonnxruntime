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
"""Define ONNX Constant operator."""
# pylint: disable=unused-argument
# pylint: disable=g-explicit-length-test
from collections.abc import Callable, Sequence
import functools
from typing import Any
from jax import jit
from jax import numpy as jnp
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node
import onnx


def _asarray(proto):
  return jnp.asarray(
      onnx.numpy_helper.to_array(proto).reshape(tuple(proto.dims))
  )


@handler.register_op('Constant')
class Constant(handler.Handler):
  """Implementation of the ONNX Constant operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    attr_to_dtype = {
        'value_int': jnp.int64,
        'value_ints': jnp.int64,
        'value_float': jnp.float32,
        'value_floats': jnp.float32,
    }

    matched = 0
    if 'value_string' in node.attrs:
      node.attrs_dict['value'] = node.attrs['value_string']
      matched = matched + 1
    elif 'value_strings' in node.attrs:
      node.attrs_dict['value'] = node.attrs['value_strings']
      matched = matched + 1
    elif 'value' in node.attrs:
      node.attrs_dict['value'] = _asarray(node.attrs['value'])
      matched = matched + 1
    else:
      for item in attr_to_dtype:
        if item in node.attrs:
          node.attrs_dict['value'] = jnp.array(
              node.attrs[item], dtype=attr_to_dtype[item]
          )
        matched = matched + 1

    assert (
        matched == 1
    ), f'Should only provide one of value attributes, but get {matched}'

  @classmethod
  def version_13(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_13 Constant op."""
    cls._prepare(node, inputs, onnx_constant)
    return onnx_constant


@functools.partial(jit, static_argnames=())
def onnx_constant(*input_args, value):
  """The impl for https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#Constant."""
  assert len(input_args) == 0
  return value
