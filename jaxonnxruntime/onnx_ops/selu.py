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

"""Define ONNX Selu operator."""
# pylint: disable=unused-argument
# pylint: disable=g-explicit-length-test
from collections.abc import Callable, Sequence
import functools
from typing import Any

import jax
from jax import jit
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node


@handler.register_op('Selu')
class Selu(handler.Handler):
  """Implementation of the ONNX Selu operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    node.attrs_dict['alpha'] = node.attrs.get(
        'alpha', 1.67326319217681884765625
    )
    node.attrs_dict['gamma'] = node.attrs.get(
        'gamma', 1.05070102214813232421875
    )

  @classmethod
  def version_6(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_6 Selu op."""
    cls._prepare(node, inputs, onnx_selu)
    return onnx_selu


@functools.partial(jit, static_argnames=('alpha', 'gamma'))
def onnx_selu(*input_args, alpha, gamma):
  """https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#Selu for more details."""
  assert len(input_args) == 1
  data = input_args[0]
  return gamma * jax.nn.elu(data, alpha)
