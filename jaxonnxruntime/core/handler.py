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

"""Defines a Handler class and a decorator to register ONNX ops."""
import inspect
import logging
from typing import Any
from onnx import defs

logger = logging.getLogger(__name__)

OnnxNode = Any
JaxFunc = Any


class Handler:
  """Base class for ONNX op."""

  DOMAIN: str = ""  # Domain of the op
  OP_TYPE: str = ""  # Type of the op
  SINCE_VERSION: int = 0  # Version since which the op is available

  @classmethod
  def get_since_version(cls, version: int) -> int:
    """Get the SINCE_VERSION based on the VERSION of the ONNX opset being used."""
    domain = cls.DOMAIN
    op_type = cls.OP_TYPE
    since_version = 1
    try:
      since_version = defs.get_schema(
          op_type,
          domain=domain,
          max_inclusive_version=version,
      ).since_version
    except Exception:  # pylint: disable=broad-except
      logger.warning(
          (
              "Fail to get since_version of %s in domain %s{domain} "
              "with max_inclusive_version= %i {version}. Set to 1."
          ),
          op_type,
          domain,
          version,
      )
    return since_version

  @classmethod
  def handle(cls, node: OnnxNode, **kwargs) -> Any:
    ver_handle = getattr(cls, "version_{}".format(cls.SINCE_VERSION), None)
    if ver_handle:
      return ver_handle(node, **kwargs)  # pylint: disable=not-callable

    raise NotImplementedError(
        "{} version {} is not implemented.".format(
            node.op_type, cls.SINCE_VERSION
        )
    )

  @classmethod
  def prepare_attrs_dict(cls, node: OnnxNode, onnx_jax_impl: JaxFunc) -> None:
    """Prepare attrs_dict for the jax function onnx_jax_impl(*inputs, **attrs_dict)."""
    sig = inspect.signature(onnx_jax_impl)
    kwparams = [param.name for param in sig.parameters.values() if param.kind == inspect.Parameter.KEYWORD_ONLY]
    node.attrs_dict = {name: node.attrs.get(name, None) for name in kwparams}

def register_op(op_type: str, domain: str = "") -> Any:
  """Register op into specific domain. default value "" is ai.onnx domain."""

  def deco(cls):
    setattr(cls, "DOMAIN", domain)
    setattr(cls, "OP_TYPE", op_type)
    return cls

  return deco
