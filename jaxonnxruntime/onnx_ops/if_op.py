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
"""Define ONNX If operator."""
# pylint: disable=unused-argument
# pylint: disable=g-explicit-length-test
from collections.abc import Callable, Sequence
from typing import Any

from jax import jit
from jax.lax import cond
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node


@handler.register_op("If")
class If(handler.Handler):
  """Implementation of the ONNX If operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    pass

  @classmethod
  def version_16(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_16 If op."""
    cls._prepare(node, inputs, onnx_if)
    return onnx_if


# The input `params` to `model_func` in `else_branch` is a dict, which is not
# hashable, and hence `else_branch` or `then_branch` cannot be set as static
# argument here. Is there a solution?
# @functools.partial(jit, static_argnames=("else_branch", "then_branch",))
def onnx_if(
    *input_args,
    else_branch,
    then_branch,
    else_branch_input_num,
    then_branch_input_num
):
  """https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#If for more details."""
  assert len(input_args) == 1 + else_branch_input_num + then_branch_input_num
  else_body, else_params = else_branch
  else_inputs = input_args[1 : 1 + else_branch_input_num]
  then_body, then_params = then_branch
  then_inputs = input_args[1 + else_branch_input_num :]
  inputs = cond(
      input_args[0], get_then_inputs, get_else_inputs, then_inputs, else_inputs
  )
  params = cond(
      input_args[0], get_then_inputs, get_else_inputs, then_params, else_params
  )
  return cond(input_args[0], then_body, else_body, params, inputs)


@jit
def get_else_inputs(_, else_inputs):
  return else_inputs


@jit
def get_then_inputs(then_inputs, _):
  return then_inputs
