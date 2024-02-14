# Copyright 2024 The Jaxonnxruntime Authors.
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

"""Utilities for creating onnx ops."""

from collections.abc import Callable
import inspect
from typing import Any

from jaxonnxruntime.core import onnx_node


def update_node_attrs_dict(
    node: onnx_node.OnnxNode, onnx_jax_impl: Callable[..., Any]
):
  """Updates the node's attrs_dict with the values from the node's attrs."""
  sig = inspect.signature(onnx_jax_impl)
  kwparams = [
      param.name
      for param in sig.parameters.values()
      if param.kind == inspect.Parameter.KEYWORD_ONLY
  ]
  for name in kwparams:
    node.attrs_dict[name] = node.attrs.get(name, None)
