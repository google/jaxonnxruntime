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
"""Wrap the onnx.GraphProto as OnnxGraph class and provide useful graph manipulation methods."""

import collections
import functools
from typing import Any, List, Sequence, Union

import jax
from jaxonnxruntime.core import onnx_node
from jaxonnxruntime.core import onnx_utils
import numpy as np

import onnx


class OnnxGraph:
  """Graph class wrapper of ONNX.GraphProto.

  Attributes:
    graph_proto: The ONNX GraphProto object.
    node_dict: A dictionary containing all the nodes in the graph, indexed by
      their names.
    initializer_dict: A dictionary containing all the initializers in the graph,
      indexed by their names.
    input: A list of input names in the graph.
    output: A list of output names in the graph.
    doc_string: The docstring of the ONNX graph.
    name: The name of the ONNX graph.
    value_info_dict: A dictionary containing all the value_info in the graph,
      indexed by their names.
    metadata: A dictionary containing the metadata of the graph.
  """

  initializer_dict: dict[str, jax.Array]
  node_dict: dict[str, onnx.NodeProto]
  input: list[str]
  output: list[str]
  doc_string: str
  name: str
  value_info_dict: dict[str, onnx.ValueInfoProto]
  metadata: dict[str, Any]

  def __init__(self, graph_proto: onnx.GraphProto):
    self.graph_proto: onnx.GraphProto = graph_proto
    self.node_dict: dict[str, onnx.NodeProto] = {}
    for index, nd in enumerate(graph_proto.node):
      node_name = f"node_{index}"
      nd.name = node_name
      self.node_dict[node_name] = nd
    self.initializer_dict: dict[str, Union[jax.Array, np.ndarray]] = {
        ts.name: onnx_utils.onnx_tensor_to_np_array(ts)
        for ts in graph_proto.initializer
    }
    self.input: list[str] = [proto.name for proto in graph_proto.input]
    self.output: list[str] = [proto.name for proto in graph_proto.output]
    self.doc_string: str = graph_proto.doc_string
    self.name: str = graph_proto.name
    self.value_info_dict: dict[str, onnx.ValueInfoProto] = {
        **{proto.name: proto for proto in graph_proto.input},
        **{proto.name: proto for proto in graph_proto.output},
        **{proto.name: proto for proto in graph_proto.value_info},
    }
    self.metadata: dict[str, Any] = {}
    self._initialize_metadata()

  @functools.lru_cache(maxsize=128)
  def get_constant_dict(self) -> dict[str, Any]:
    """Get a dictionary of constant tensors."""
    results = dict()
    for node in self.graph_proto.node:
      if node.op_type == "Constant":
        node_wrapper = onnx_node.OnnxNode(node)
        results[node.output[0]] = node_wrapper.get_constant_node_value()
    return results

  def _initialize_metadata(self):
    """Initialize the meta_data dict."""
    # Build those dicts link tensors and nodes
    # key is tensor, value is the list of node who take this tensor as input.
    tensor_down_to_node_dict = {}
    # key is tensor, value is the node who output this tensor.
    tensor_up_to_node_dict = {}
    for nd_name, nd in self.node_dict.items():
      for input_name in nd.input:
        if input_name not in tensor_down_to_node_dict:
          tensor_down_to_node_dict[input_name] = []
        tensor_down_to_node_dict[input_name].append(nd_name)

      for output_name in nd.output:
        tensor_up_to_node_dict[output_name] = nd_name

    # Build the node_to_tensor_dict
    node_down_to_tensor_dict = {}
    node_up_to_tensor_dict = {}
    for nd_name, nd in self.node_dict.items():
      output_names = list(nd.output)
      node_down_to_tensor_dict[nd_name] = output_names
      input_names = list(nd.input)
      node_up_to_tensor_dict[nd_name] = input_names

    # This dictionary maps the input tensor to those nodes that comsume it.
    self.metadata["tensor_down_to_node_dict"]: dict[
        str, list[str]
    ] = tensor_down_to_node_dict
    # This dictionary maps the output tensor to the node that produce them.
    self.metadata["tensor_up_to_node_dict"]: dict[
        str, str
    ] = tensor_up_to_node_dict
    # This dictionary maps the node to those output tensors by this node.
    self.metadata["node_down_to_tensor_dict"]: dict[
        str, list[str]
    ] = node_down_to_tensor_dict
    # This dictionary maps the node to those input tensors of this node.
    self.metadata["node_up_to_tensor_dict"]: dict[
        str, list[str]
    ] = node_up_to_tensor_dict

  def get_real_input(self) -> list[str]:
    """Returns unique non-node input names."""
    real_input: list[str] = []
    output_list: list[str] = []
    for node in self.node_dict.values():
      output_list.extend(node.output)
    for node in self.node_dict.values():
      real_input.extend(
          i
          for i in node.input
          if i not in self.initializer_dict and i not in output_list
      )

    # Sometime input name is empty string(""), should be removed.
    # Also we need remove dumplicate
    unique_real_input = []
    for item in real_input:
      if item not in unique_real_input and item != "":  # pylint: disable=g-explicit-bool-comparison
        unique_real_input.append(item)
    return unique_real_input

  def get_parent_nodes_name(self, node_name: str) -> List[str]:
    """Get the names of the parent nodes of a given node."""
    node_up_to_tensor_dict = self.metadata["node_up_to_tensor_dict"]
    tensor_up_to_node_dict = self.metadata["tensor_up_to_node_dict"]
    assert node_name in node_up_to_tensor_dict
    inputs = node_up_to_tensor_dict[node_name]
    return [
        tensor_up_to_node_dict[i] for i in inputs if i in tensor_up_to_node_dict
    ]

  def get_tensor_parent_node_name(self, tensor_name: str) -> str:
    """Get the name of the parent node of a given tensor."""
    tensor_up_to_node_dict = self.metadata["tensor_up_to_node_dict"]
    assert tensor_name in tensor_up_to_node_dict
    return tensor_up_to_node_dict[tensor_name]

  def get_value_info_shape(self, tensor_name: str) -> list[Union[str, int]]:
    """Extracts the shape of an ONNX ValueInfoProto as a list.

    Args:
      tensor_name: The ONNX tensor name.

    Returns:
      list: A list representing the shape of the ValueInfoProto.
    """
    assert tensor_name in self.value_info_dict
    value_info_proto = self.value_info_dict[tensor_name]
    shape = []
    for dim in value_info_proto.type.tensor_type.shape.dim:
      if dim.HasField("dim_value"):
        shape.append(dim.dim_value)
      elif dim.HasField("dim_param"):
        shape.append(dim.dim_param)
    return shape

  def get_child_nodes_name(self, node_name: str) -> List[str]:
    """Get the names of the children nodes of a given node."""
    node_down_to_tensor_dict = self.metadata["node_down_to_tensor_dict"]
    tensor_down_to_node_dict = self.metadata["tensor_down_to_node_dict"]
    assert node_name in node_down_to_tensor_dict
    outputs = node_down_to_tensor_dict[node_name]
    results = []
    for output_ in outputs:
      if output_ and output_ in tensor_down_to_node_dict:
        # Ignore empty strings as they represent unused optional outputs.
        results.extend(tensor_down_to_node_dict[output_])
    return results

  def get_tensor_child_node_name(self, tensor_name: str) -> List[str]:
    """Get the names of the children nodes of a given tensor."""
    tensor_down_to_node_dict = self.metadata["tensor_down_to_node_dict"]
    assert tensor_name in tensor_down_to_node_dict
    return tensor_down_to_node_dict[tensor_name]

  def topological_sort(self) -> Sequence[onnx.NodeProto]:
    """Return the topological sort order of those nodes."""
    in_degree = collections.defaultdict(int)
    for u in self.node_dict:
      if u not in in_degree:
        in_degree[u] = 0
      for v in self.get_child_nodes_name(u):
        in_degree[v] += 1

    queue = collections.deque(
        [v for v in in_degree.keys() if in_degree[v] == 0]
    )
    sorted_list = []

    while queue:
      u = queue.popleft()
      sorted_list.append(u)

      for v in self.get_child_nodes_name(u):
        in_degree[v] -= 1
        if in_degree[v] == 0:
          queue.append(v)

    if len(sorted_list) != len(self.node_dict):
      raise RuntimeError(
          f"Graph has a cycle, sorted_list={sorted_list},"
          f" node_dict={self.node_dict.keys()}"
      )

    return [self.node_dict[n] for n in sorted_list]
