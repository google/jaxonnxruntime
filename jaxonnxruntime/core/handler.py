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
import time
from typing import Any
import jax
from jaxonnxruntime.core import config_class
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
    config = config_class.config

    onnx_impl = None
    if config.jaxort_use_pallas:
      pallas_handle = getattr(
          cls, "pallas_version_{}".format(cls.SINCE_VERSION), None
      )
      if pallas_handle:
        onnx_impl = pallas_handle(node, inputs, **kwargs)  # pylint: disable=not-callable

    if onnx_impl is None:
      ver_handle = getattr(cls, "version_{}".format(cls.SINCE_VERSION), None)
      if ver_handle:
        onnx_impl = ver_handle(node, inputs, **kwargs)  # pylint: disable=not-callable

    if onnx_impl is None:
      # Get all the methods that start with "version_"
      class_methods = inspect.getmembers(cls, predicate=inspect.ismethod)
      version_methods = [
          method_name
          for method_name, _ in class_methods
          if method_name.startswith("version_")
      ]

      raise NotImplementedError(
          f"{node.op_type} version {cls.SINCE_VERSION} is not implemented."
          f" Only have those versions: {version_methods}."
      )

    if config.jaxort_benchmark:

      def benchmark_wrapped(*args, **kwargs):
        # Warmup
        _ = onnx_impl(*args, **kwargs)

        start = time.time()
        for _ in range(100):
          res = onnx_impl(*args, **kwargs)
          jax.block_until_ready(res)
        end = time.time()
        mean_ms = (end - start) / 100 * 1000
        print(f"Op {cls.OP_TYPE} mean time over 100 runs: {mean_ms:.3f} ms")
        return onnx_impl(*args, **kwargs)

      return benchmark_wrapped

    return onnx_impl

  @classmethod
  def _prepare(
      cls,
      node: OnnxNode,
      inputs: Sequence[Any],
      onnx_jax_impl: Callable[..., Any],
  ) -> None:
    """Rwrite the OnnxNode to prepare the inputs attributes for the onnx jax implementation."""
    raise NotImplementedError


def register_op(op_type: str, domain: str = "") -> Callable[[OnnxOp], OnnxOp]:
  """Register op into specific domain. default value "" is ai.onnx domain."""

  def deco(cls: OnnxOp):
    setattr(cls, "DOMAIN", domain)
    setattr(cls, "OP_TYPE", op_type)
    return cls

  return deco
