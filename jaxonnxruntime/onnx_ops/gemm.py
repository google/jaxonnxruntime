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
"""Define ONNX Gemm operator."""
from collections.abc import Callable, Sequence
import functools
import inspect
from typing import Any, Optional

from jax import jit
from jax import numpy as jnp
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node


@handler.register_op('Gemm')
class Gemm(handler.Handler):
  """Implementation of the ONNX Gemm operator."""

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

  @classmethod
  def version_13(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_13 Gemm op."""
    cls._prepare(node, inputs, onnx_gemm)
    return onnx_gemm


@functools.partial(jit, static_argnames=('alpha', 'beta', 'transA', 'transB'))
def onnx_gemm(
    *input_args,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    transA: Optional[int] = None,
    transB: Optional[int] = None
):
  """The impl for https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#Gemm."""
  assert len(input_args) == 3 or len(input_args) == 2
  if len(input_args) == 2:
    a, b = input_args
    c = 0
  else:
    a, b, c = input_args

  alpha = 1.0 if not alpha else alpha
  beta = 1.0 if not beta else beta
  transA = 0 if not transA else transA
  transB = 0 if not transB else transB

  if transA == 1:
    a = jnp.transpose(a)
  if transB == 1:
    b = jnp.transpose(b)

    # Compute the matrix multiplicat
  return alpha * jnp.dot(a, b) + beta * c
