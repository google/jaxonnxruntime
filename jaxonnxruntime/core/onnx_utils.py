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
"""onnx utility functions collection."""
from jax import numpy as jnp
import onnx


def tensor_dtype_to_jnp_dtype(
    tensor_type: onnx.TensorProto.DataType,
) -> jnp.dtype:
  """Convert onnx.TensorProto.DataType to jnp.dtype."""
  if tensor_type is onnx.TensorProto.BFLOAT16:
    return jnp.bfloat16
  if onnx.__version__ < '1.14.0':
    np_type = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[tensor_type]
  else:
    np_type = onnx.helper.tensor_dtype_to_np_dtype(tensor_type)
  return jnp.dtype(np_type)


def get_shape_and_dtype_from_val_info(
    value_info: onnx.ValueInfoProto,
) -> tuple[list[int], jnp.dtype]:
  """Get jax numpy shape and dtype from onnx.ValueInfoProto."""
  type_proto = value_info.type
  dtype = tensor_dtype_to_jnp_dtype(type_proto.tensor_type.elem_type)
  shape = [dim.dim_value for dim in type_proto.tensor_type.shape.dim]

  return shape, dtype
