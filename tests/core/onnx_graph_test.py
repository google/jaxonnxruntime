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

from absl.testing import absltest
from jaxonnxruntime.core import onnx_graph

import onnx

OnnxGraph = onnx_graph.OnnxGraph


class TestOnnxGraph(absltest.TestCase):

  def setUp(self):
    # create a simple ONNX graph proto with an Add and Relu node
    super().setUp()
    graph_proto = onnx.GraphProto(
        input=[onnx.ValueInfoProto(name="x")],
        output=[onnx.ValueInfoProto(name="y")],
        node=[
            onnx.NodeProto(
                name="node_0",
                op_type="Add",
                input=["x", "x"],
                output=["add_out"],
            ),
            onnx.NodeProto(
                name="node_1",
                op_type="Conv",
                input=["add_out", "weight"],
                output=["conv_out"],
            ),
            onnx.NodeProto(
                name="node_2",
                op_type="Relu",
                input=["conv_out"],
                output=["y"],
            ),
        ],
    )
    self.graph = OnnxGraph(graph_proto)

  def test_get_real_input(self):
    real_input = self.graph.get_real_input()
    self.assertEqual(real_input, ["x", "weight"])

  def test_get_parent_nodes_name(self):
    parent_nodes = self.graph.get_parent_nodes_name("node_1")
    self.assertEqual(parent_nodes, ["node_0"])

  def test_get_child_nodes_name(self):
    child_nodes = self.graph.get_child_nodes_name("node_1")
    self.assertEqual(child_nodes, ["node_2"])

  def test_topological_sort(self):
    node_order = self.graph.topological_sort()
    self.assertLen(node_order, 3)
    self.assertEqual(node_order[0].op_type, "Add")
    self.assertEqual(node_order[1].op_type, "Conv")
    self.assertEqual(node_order[2].op_type, "Relu")


if __name__ == "__main__":
  absltest.main()
