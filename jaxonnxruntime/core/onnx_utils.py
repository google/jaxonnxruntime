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
import jax
from jax import numpy as jnp
import onnx
from onnx import numpy_helper


def tensor_dtype_to_jnp_dtype(
    tensor_type: onnx.TensorProto.DataType,
) -> jnp.dtype:
  """Convert onnx.TensorProto.DataType to jnp.dtype."""
  if tensor_type is onnx.TensorProto.BFLOAT16:
    return jnp.bfloat16
  if onnx.__version__ < "1.14.0":
    np_type = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[tensor_type]
  else:
    np_type = onnx.helper.tensor_dtype_to_np_dtype(tensor_type)
  return jnp.dtype(np_type)


def get_elem_type_from_type_proto(type_proto: onnx.TypeProto):
  if type_proto.HasField("optional_type"):
    return get_elem_type_from_type_proto(type_proto.optional_type.elem_type)
  if type_proto.HasField("sequence_type"):
    return get_elem_type_from_type_proto(type_proto.sequence_type.elem_type)

  if type_proto.HasField("tensor_type"):
    return type_proto.tensor_type.elem_type

  raise ValueError(
      f"currently only support Tensor type TypeProto but got {type_proto}"
  )


def get_shape_and_dtype_from_val_info(
    value_info: onnx.ValueInfoProto,
) -> tuple[list[int], jnp.dtype]:
  """Get jax numpy shape and dtype from onnx.ValueInfoProto."""
  type_proto = value_info.type
  elem_type = get_elem_type_from_type_proto(type_proto)
  dtype = tensor_dtype_to_jnp_dtype(elem_type)
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


def valueinfoproto_asarray(proto: Any) -> jax.Array:
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


def sanitize_tensor_names_in_graph(
    graph: onnx.GraphProto,
) -> onnx.GraphProto:
  """Format the names of all tensors in an onnx.GraphProto.

  Each tensors will have a unique name in the format 'tensor_{idx}'.
  Args:
    graph: the onnx.GraphProto to be processed.

  Returns:
    graph: the graph within which tensor names have been formatted.
  """

  def _unique_tensor_name_generator():
    idx = 0
    while True:
      yield f"tensor_{str(idx)}"
      idx += 1

  unique_name_gen = _unique_tensor_name_generator()
  name_map = {}

  def _sanitize_tensor_names_in_graph(graph):
    for nd in graph.node:
      for i in range(len(nd.input)):
        if nd.input[i] not in name_map:
          name_map[nd.input[i]] = next(unique_name_gen)
        nd.input[i] = name_map[nd.input[i]]
      for i in range(len(nd.output)):
        if nd.output[i] not in name_map:
          name_map[nd.output[i]] = next(unique_name_gen)
        nd.output[i] = name_map[nd.output[i]]
      if contain_subgraph(nd):
        for attr_proto in nd.attribute:
          if attr_proto.HasField("g"):
            _sanitize_tensor_names_in_graph(attr_proto.g)
    for proto in graph.initializer:
      if proto.name not in name_map:
        name_map[proto.name] = next(unique_name_gen)
      proto.name = name_map[proto.name]
    for proto in graph.input:
      if proto.name not in name_map:
        name_map[proto.name] = next(unique_name_gen)
      proto.name = name_map[proto.name]
    for proto in graph.output:
      if proto.name not in name_map:
        name_map[proto.name] = next(unique_name_gen)
      proto.name = name_map[proto.name]

  _sanitize_tensor_names_in_graph(graph)
  return graph
