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
import functools
from typing import Any

from jax import jit
from jax import numpy as jnp
from jax.lax import cond
from jaxonnxruntime import config
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node
from jaxonnxruntime.core import onnx_utils

import onnx


@handler.register_op("If")
class If(handler.Handler):
  """Implementation of the ONNX If operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    flatten_subgraph(node, inputs)

  @classmethod
  def version_16(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_16 If op."""
    cls._prepare(node, inputs, onnx_if)
    return onnx_if


def flatten_subgraph(node, inputs):
  """Recursively construct the subgraphs for else and then branches."""
  from jaxonnxruntime.call_onnx import call_onnx_graph  # pylint: disable=g-import-not-at-top

  inp_start = 1
  for a_name, a in node.attrs.items():
    if isinstance(a, onnx.GraphProto):
      input_names = onnx_utils.get_graph_input(a)
      inp_end = inp_start + len(input_names)
      subgraph_inps = inputs[inp_start:inp_end]
      params = {
          n.name: onnx_utils.valueinfoproto_asarray(n) for n in a.initializer
      }
      tensor_dict = dict(
          **onnx_utils.maybe_convert_to_dict(subgraph_inps, input_names),
          **params,
      )
      jax_func = call_onnx_graph(a, tensor_dict)
      del tensor_dict
      node.attrs_dict[a_name] = jax_func
      node.attrs_dict[f"{a_name}_params"] = params
      node.attrs_dict[f"{a_name}_input_num"] = len(input_names)
      inp_start = inp_end


@functools.partial(
    jit,
    static_argnames=(
        "else_branch",
        "then_branch",
        "else_branch_input_num",
        "then_branch_input_num",
    ),
)
def onnx_if(
    *input_args,
    else_branch_params,
    then_branch_params,
    else_branch,
    then_branch,
    else_branch_input_num,
    then_branch_input_num,
):
  """https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#If for more details."""
  assert len(input_args) == 1 + else_branch_input_num + then_branch_input_num
  else_inputs = input_args[1 : 1 + else_branch_input_num]
  then_inputs = input_args[1 + else_branch_input_num :]

  @jit
  def run_else_branch(y, z, else_branch_params, else_inputs):
    else_result = else_branch(else_branch_params, else_inputs)
    if config.jaxort_if_op_reshape_output_for_llama:
      return [jnp.squeeze(else_result[0], axis=0)]
    else:
      return else_result

  @jit
  def run_then_branch(then_branch_params, then_inputs, y, x):
    then_result = then_branch(then_branch_params, then_inputs)
    return then_result

  res = cond(
      jnp.squeeze(input_args[0]),
      run_then_branch,
      run_else_branch,
      then_branch_params,
      then_inputs,
      else_branch_params,
      else_inputs,
  )
  return res
