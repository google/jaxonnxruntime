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
"""Define Custom ONNX ZerosLike operator.

Here we demo how to reuse Tensorflow ops impelmentation.
"""


from collections.abc import Callable, Sequence
import functools
import inspect
from typing import Any

import jax
from jax import jit
from jax.experimental import jax2tf
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node
from jaxonnxruntime.core import onnx_utils
import tensorflow as tf


@handler.register_op("ZerosLike", domain="jaxonnxruntime")
class ZerosLike(handler.Handler):
  """Implementation of the ONNX ZerosLike custom operator."""

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
      node.attrs_dict[name] = node.attrs.get(name)

  @classmethod
  def version_1(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_1 Identity op."""
    cls._prepare(node, inputs, tf_zeros_like)
    return tf_zeros_like


@functools.partial(jit, static_argnames="dtype")
def tf_zeros_like(x: jax.Array, *, dtype: int):
  """https://www.tensorflow.org/api_docs/python/tf/zeros_like for more details."""
  jax_dtype = onnx_utils.tensor_dtype_to_jnp_dtype(dtype)

  def tf_func(input0):
    return tf.zeros_like(input0, dtype=jax_dtype)

  jax_func = jax2tf.call_tf(tf_func)
  return jax_func(x)
