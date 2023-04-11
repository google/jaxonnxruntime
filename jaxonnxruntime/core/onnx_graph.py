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
from typing import Any, List, Sequence
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

  def __init__(self, graph_proto: onnx.GraphProto):
    self.graph_proto = graph_proto
    self.node_dict = {}
    for index, nd in enumerate(graph_proto.node):
      node_name = f"node_{index}"
      nd.name = node_name
      self.node_dict[node_name] = nd
    self.initializer_dict = {ts.name: ts for ts in graph_proto.initializer}
    self.input = [proto.name for proto in graph_proto.input]
    self.output = [proto.name for proto in graph_proto.output]
    self.doc_string = graph_proto.doc_string
    self.name = graph_proto.name
    self.value_info_dict = {
        **{proto.name: proto for proto in graph_proto.input},
        **{proto.name: proto for proto in graph_proto.output},
        **{proto.name: proto for proto in graph_proto.value_info},
    }
    self.metadata: dict[str, Any] = {}
    self._initialize_metadata()

  def _initialize_metadata(self):
    """Initialize the meta_data dict."""
    # Build those dicts link tensors and nodes
    tensor_down_to_node_dict = {}
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

  def get_child_nodes_name(self, node_name: str) -> List[str]:
    """Get the names of the children nodes of a given node."""
    node_down_to_tensor_dict = self.metadata["node_down_to_tensor_dict"]
    tensor_down_to_node_dict = self.metadata["tensor_down_to_node_dict"]
    assert node_name in node_down_to_tensor_dict
    outputs = node_down_to_tensor_dict[node_name]
    results = []
    for output_ in outputs:
      if output_ in tensor_down_to_node_dict:
        results.extend(tensor_down_to_node_dict[output_])
    return results

  def topological_sort(self) -> Sequence[onnx.NodeProto]:
    """Return the topological sort order of those nodes."""

    visited = {}
    stack = []

    # A recursive function used by topologicalSort
    def topological_sort_util(v):
      visited[v] = True
      for i in self.get_child_nodes_name(v):
        if i not in visited or not visited[i]:
          topological_sort_util(i)
      stack.append(v)

    for i in self.node_dict:
      if i not in visited or not visited[i]:
        topological_sort_util(i)

    # return list in reverse order.
    return list(reversed([self.node_dict[n] for n in stack]))
