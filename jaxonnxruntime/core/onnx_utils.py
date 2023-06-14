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
from typing import Any, Dict, Optional, Sequence, Union
from jax import numpy as jnp
import onnx
from onnx import numpy_helper


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


def contain_subgraph(node: Any) -> bool:
  """Check if the node contains subgraph (control flow)."""
  return node.op_type in ("If", "Loop")


def get_graph_input(graph: onnx.GraphProto) -> list[str]:
  """Returns unique non-node input names."""
  real_input: list[str] = []
  output_list: list[str] = []
  initializers: list[str] = [ts.name for ts in graph.initializer]
  for node in graph.node:
    output_list.extend(list(node.output))
  for node in graph.node:
    real_input.extend(
        i for i in node.input if i not in initializers and i not in output_list
    )

  # Sometimes input name is empty string(""), which should be removed.
  # We also need to remove duplicates.
  unique_real_input = []
  for item in real_input:
    if item not in unique_real_input and item != "":  # pylint: disable=g-explicit-bool-comparison
      unique_real_input.append(item)
  return unique_real_input


def valueinfoproto_asarray(proto: Any):
  """Convert onnx.ValueInfoProto to jaxlib.xla_extension.ArrayImpl."""
  return jnp.asarray(numpy_helper.to_array(proto).reshape(tuple(proto.dims)))


def maybe_convert_to_dict(
    inputs: Union[Sequence[Any], Dict[str, Any]],
    input_names: Optional[Sequence[Any]] = None,
):
  """Convert inputs to a dictionary with input_names as keys."""
  if isinstance(inputs, dict):
    return inputs
  elif isinstance(inputs, Sequence):
    if input_names is None:
      raise ValueError("Should provide input names if `inputs` is a Sequence!")
    assert len(inputs) == len(input_names)
    return dict(zip(input_names, inputs))
  else:
    raise NotImplementedError("Please use inputs of type dict or Sequence!")
