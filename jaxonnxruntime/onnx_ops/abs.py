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
"""Define ONNX Abs operator."""
# pylint: disable=unused-argument
# pylint: disable=g-explicit-length-test
from collections.abc import Callable, Sequence
import functools
import inspect
from typing import Any

import jax
from jax import jit
from jax import numpy as jnp
from jax._src.interpreters import mlir
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node
from jaxonnxruntime.core import onnx_primitive


@handler.register_op("Abs")
class Abs(handler.Handler):
  """Implementation of the ONNX Abs operator."""

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
    """ONNX version 13 Abs op."""
    cls._prepare(node, inputs, onnx_abs)
    return onnx_abs


@functools.partial(jit, static_argnames=())
def onnx_abs(*args):
  """https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#Abs for more details."""

  if len(args) != 1:
    raise ValueError(
        f"len(args) should equal to 1 but got {len(args)}"
    )
  all_args = args

  return onnx_abs_p.bind(*all_args)

# Define onnx_abs_p primitive.
onnx_abs_p = onnx_primitive.OnnxPrimitive("onnx_abs")
onnx_abs_p.multiple_results = False


@onnx_abs_p.def_impl
def _onnx_abs_impl(*args):
  x = args[0]
  return jnp.abs(x)


@onnx_abs_p.def_abstract_eval
def _onnx_abs_abstract_eval(*args):
  aval_args = jax.tree_map(
      lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), args
  )
  out = jax.eval_shape(_onnx_abs_impl, *aval_args)
  return jax.tree_map(
      lambda x: jax.abstract_arrays.ShapedArray(x.shape, x.dtype), out
  )


def _onnx_abs_lowering(ctx, *args, platform):
  """abs lowering rule."""
  jit_func = jax.jit(_onnx_abs_impl)
  jit_func_lowering = mlir.lower_fun(jit_func, onnx_abs_p.multiple_results)
  return mlir.delegate_lowering(ctx, jit_func_lowering, *args)


for _p in ("cpu", "tpu", "cuda", "rocm"):
  mlir.register_lowering(
      onnx_abs_p,
      functools.partial(_onnx_abs_lowering, platform=_p),
      platform=_p,
  )
