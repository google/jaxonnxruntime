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
"""Define ONNX ConstantOfShape operator."""
# pylint: disable=unused-argument
from collections.abc import Callable, Sequence
import functools
import inspect
from typing import Any
from jax import jit
from jax import numpy as jnp
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node
import onnx


@handler.register_op('ConstantOfShape')
class ConstantOfShape(handler.Handler):
  """Implementation of the ONNX ConstantOfShape operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    sig = inspect.signature(onnx_jax_impl)
    kwparams = [
        param.name
        for param in sig.parameters.values()
        if param.kind == inspect.Parameter.KEYWORD_ONLY
    ]
    for name in kwparams:
      node.attrs_dict[name] = node.attrs.get(name, None)
    assert len(inputs) == 1
    node.attrs_dict['shape'] = tuple(inputs[0].tolist())
    if 'value' in node.attrs_dict:
      np_value = onnx.numpy_helper.to_array(node.attrs_dict['value'])
      if len(np_value.tolist()) != 1:
        raise ValueError(
            'ONNX ConstantOfShape op `value` attr should contain only 1 value'
            f' but got {np_value} on node {node.node_proto}'
        )
      node.attrs_dict['value'] = np_value.tolist()[0]
      node.attrs_dict['dtype'] = np_value.dtype

  @classmethod
  def version_9(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_9 ConstantOfShape op."""
    cls._prepare(node, inputs, onnx_constantofshape)
    return onnx_constantofshape


@functools.partial(jit, static_argnames=('value', 'shape', 'dtype'))
def onnx_constantofshape(*input_args, value=0, shape=None, dtype=jnp.float32):
  """The internal jax impl for onnx ConstantOfShape op."""
  return jnp.full(fill_value=value, shape=shape, dtype=dtype)
