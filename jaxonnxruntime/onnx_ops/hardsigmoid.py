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
"""Define ONNX HardSigmoid operator."""
from collections.abc import Callable, Sequence
import functools
import inspect
from typing import Any

import jax
from jax import jit
from jax import numpy as jnp
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node


@handler.register_op("HardSigmoid")
class HardSigmoid(handler.Handler):
  """Implementation of the ONNX HardSigmoid operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ) -> None:
    sig = inspect.signature(onnx_jax_impl)
    kwparams = [
        param.name
        for param in sig.parameters.values()
        if param.kind == inspect.Parameter.KEYWORD_ONLY
    ]
    for name in kwparams:
      node.attrs_dict[name] = node.attrs.get(name, None)

  @classmethod
  def version_6(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_6 HardSigmoid op."""
    cls._prepare(node, inputs, onnx_hardsigmoid)
    return onnx_hardsigmoid


@functools.partial(jit, static_argnames=("alpha", "beta"))
def onnx_hardsigmoid(
    *input_args: jax.Array, alpha: float = 0.2, beta: float = 0.5
) -> jax.Array:
  """The internal jax impl for onnx HardSigmoid op."""
  assert len(input_args) == 1
  x = input_args[0]
  if alpha is None:
    alpha = 0.2
  if beta is None:
    beta = 0.5
  return jnp.clip(alpha * x + beta, 0.0, 1.0)
