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

"""jax.export.Exported utils."""

import os
from typing import Any

import jax
from jax import export as jax_export
from jax import numpy as jnp
from jaxlib.mlir import ir
from mlir.dialects import stablehlo
import tensorflow as tf
import torch

MLIRModule = ir.Module
HloSharding = Any | None
Sharding = jax.sharding.Sharding | None


def _get_stablehlo_version() -> str:
  """Returns >=4w StableHLO version."""
  # Require 4w forward compatibility, similar to jax export.
  # See https://github.com/openxla/stablehlo/blob/main/docs/compatibility.md.
  return stablehlo.get_version_from_compatibility_requirement(
      stablehlo.StablehloCompatibilityRequirement.WEEK_4
  )


def save_exported(exp: jax_export.Exported, export_path: str) -> None:
  """Saves exported model to a file."""
  tf.io.gfile.makedirs(export_path)
  jax_exported_file = os.path.join(export_path, "jax_exported.bin")
  with tf.io.gfile.GFile(jax_exported_file, "wb") as f:
    f.write(exp.serialize(vjp_order=0))


def load_exported(export_path: str) -> jax_export.Exported:
  """Loads exported model from a file."""
  jax_exported_file = os.path.join(export_path, "jax_exported.bin")
  assert tf.io.gfile.exists(jax_exported_file)
  with tf.io.gfile.GFile(jax_exported_file, "rb") as f:
    exp = jax_export.deserialize(bytearray(f.read()))
    return exp


def tf_dtype_to_jax_dtype(tf_dtype: tf.DType):
  """Converts a tf.DType to a jax.DType."""
  return jax.dtypes.canonicalize_dtype(tf_dtype.as_numpy_dtype)


def torch_dtype_to_jax_dtype(torch_dtype: torch.dtype):
  """Converts a torch.dtype to a jax.dtype."""
  dtype_mapping = {
      torch.float32: jnp.float32,
      torch.float64: jnp.float64,
      torch.float16: jnp.float16,
      torch.int8: jnp.int8,
      torch.int16: jnp.int16,
      torch.int32: jnp.int32,
      torch.int64: jnp.int64,
      torch.uint8: jnp.uint8,
      torch.bool: jnp.bool_,
  }
  return dtype_mapping.get(torch_dtype, None)


def tf_tensor_to_jax_abstract_shaped_array(
    tf_tensor: tf.Tensor, weak_type: bool = True
) -> jax.core.ShapedArray:
  """Converts a Tensorflow tensor to a Jax ShapedArray."""
  return jax.core.ShapedArray(
      shape=list(tf_tensor.shape),
      dtype=tf_dtype_to_jax_dtype(tf_tensor.dtype),
      weak_type=weak_type,
  )


def torch_tensor_to_jax_abstract_shaped_array(
    tensor: torch.Tensor, weak_type: bool = True
) -> jax.core.ShapedArray:
  """Converts a Tensorflow tensor to a Jax ShapedArray."""
  return jax.core.ShapedArray(
      shape=list(tensor.shape),
      dtype=torch_dtype_to_jax_dtype(tensor.dtype),
      weak_type=weak_type,
  )


def torch_tensor_to_jax_array(
    tensor: torch.Tensor, inplace: bool = False
) -> jax.Array:
  """Convert a torch tensor to a jax array."""
  if not inplace:
    tensor = tensor.clone().detach()
  return jax.dlpack.from_dlpack(tensor)


def serialize_stablehlo_mlir_str(mlir_str: bytes | str) -> bytes:
  """Serializes a StableHLO MLIR module string to a bytecode."""
  target_version = _get_stablehlo_version()
  return stablehlo.serialize_portable_artifact_str(mlir_str, target_version)


def serialize_stablehlo_mlir_module(mlir_module: ir.Module) -> bytes:
  """Serializes a StableHLO MLIR module to a bytecode."""
  target_version = _get_stablehlo_version()
  mixed_serialization_ok = "1.11.0"
  allow_other_dialects = (
      stablehlo.get_smaller_version(target_version, mixed_serialization_ok)
      == mixed_serialization_ok
  )
  return stablehlo.serialize_portable_artifact(
      mlir_module, target_version, allow_other_dialects
  )
