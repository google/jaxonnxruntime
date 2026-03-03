# Copyright 2026 The Jaxonnxruntime Authors.
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

"""Defines a Handler class and a decorator to register ONNX ops."""

from collections.abc import Callable, Sequence
import inspect
import logging
from typing import Any

from onnx import defs


logger = logging.getLogger(__name__)

OnnxNode = Any
OnnxOp = Any


class Handler:
  """Base class for ONNX op."""

  DOMAIN: str = ""  # Domain of the op
  OP_TYPE: str = ""  # Type of the op
  SINCE_VERSION: int = 0  # Version since which the op is available

  @classmethod
  def get_since_version(cls, version: int) -> int:
    """Get the `since_version` of the op.

    `since_version` is the first opset version this op was added.
    Args:
      version: The version of the opset.

    Returns:
      The since version of the op.
    """
    domain = cls.DOMAIN
    op_type = cls.OP_TYPE
    try:
      since_version = defs.get_schema(
          op_type,
          domain=domain,
          max_inclusive_version=version,
      ).since_version
    except Exception:  # pylint: disable=broad-except
      # For standard onnx opset, exclude it by returning -1
      if not domain:
        return -1
      # For custom domain, return version 1.
      else:
        return 1
    return since_version

  @classmethod
  def handle(
      cls, node: OnnxNode, inputs: Sequence[Any], **kwargs
  ) -> Callable[..., Any]:
    """Return the version method jax function depending on OnnxNode verison.

    For example, onnx abs op with version 4 will call jax class `abs.version_4`
    API.

    Args:
      node: The onnx node to be handled.
      inputs: The inputs of the onnx node.
      **kwargs: The kwargs to be passed to the jax function.

    Returns:
      The jax function.
    """
    # Get all the methods that start with "version_"
    class_methods = inspect.getmembers(cls, predicate=inspect.ismethod)
    version_methods = {
        int(method_name.split("_")[1]): method
        for method_name, method in class_methods
        if method_name.startswith("version_")
    }

    if not version_methods:
      raise NotImplementedError(
          f"{node.op_type} has no versioned implementations."
      )

    # Find the largest version which is <= cls.SINCE_VERSION
    available_versions = sorted(version_methods.keys(), reverse=True)
    for v in available_versions:
      if v <= cls.SINCE_VERSION:
        return version_methods[v](node, inputs, **kwargs)

    raise NotImplementedError(
        f"{node.op_type} version {cls.SINCE_VERSION} is not implemented."
        f" Only have those versions: {sorted(version_methods.keys())}."
    )

  @classmethod
  def _prepare(
      cls,
      node: OnnxNode,
      inputs: Sequence[Any],
      onnx_jax_impl: Callable[..., Any],
  ) -> None:
    """Rwrite the OnnxNode to prepare the inputs attributes.

    Args:
      node: The onnx node to be handled.
      inputs: The inputs of the onnx node.
      onnx_jax_impl: The jax function for the onnx op.
    """
    raise NotImplementedError


def register_op(op_type: str, domain: str = "") -> Callable[[OnnxOp], OnnxOp]:
  """Register op into specific domain. default value "" is ai.onnx domain."""

  def deco(cls: OnnxOp):
    setattr(cls, "DOMAIN", domain)
    setattr(cls, "OP_TYPE", op_type)
    return cls

  return deco
