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
"""Define ONNX Add operator."""
from jax import jit
from jax import lax
from jax import numpy as jnp

from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node

register_op = handler.register_op
Handler = handler.Handler
OnnxNode = onnx_node.OnnxNode


@register_op("Add")
class Add(Handler):
  """Implementation of the ONNX Add operator."""

  @classmethod
  def version_14(cls, node, **kwargs):
    return onnx_add


@jit
def onnx_add(a, b):
  return jnp.add(a, b)
