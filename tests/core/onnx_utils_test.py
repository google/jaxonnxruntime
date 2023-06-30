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
from jaxonnxruntime.core import onnx_utils

import onnx


class TestOnnxUtils(absltest.TestCase):

  def test_sanitize_tensor_names_in_graph(self):
    else_branch = onnx.GraphProto(
        input=[onnx.ValueInfoProto(name="else_in")],
        output=[onnx.ValueInfoProto(name="else_out")],
        node=[
            onnx.NodeProto(
                name="node_0",
                op_type="Identity",
                input=["else_in"],
                output=["else_out"],
            ),
        ],
    )
    else_attr = onnx.AttributeProto(g=else_branch)
    graph = onnx.GraphProto(
        input=[onnx.ValueInfoProto(name="x")],
        initializer=[
            onnx.TensorProto(name="else_in"),
        ],
        output=[onnx.ValueInfoProto(name="y")],
        node=[
            onnx.NodeProto(
                name="node_1",
                op_type="If",
                input=["x"],
                output=["y"],
                attribute=[else_attr],  # Omit then branch for simplicity
            ),
        ],
    )
    onnx_utils.sanitize_tensor_names_in_graph(graph)
    # Tensor names change:
    # x -> tensor_0, y -> tensor_1, else_in -> tensor_2, else_out -> tensor_3
    # Graph inputs & initializers & outputs
    self.assertEqual(graph.input[0].name, "tensor_0")
    self.assertEqual(graph.initializer[0].name, "tensor_2")
    self.assertEqual(graph.output[0].name, "tensor_1")
    # Node inputs & outputs
    self.assertEqual(graph.node[0].input[0], "tensor_0")
    self.assertEqual(graph.node[0].output[0], "tensor_1")
    # Subgraph inputs & outputs & nodes
    subgraph = graph.node[0].attribute[0].g
    self.assertEqual(subgraph.input[0].name, "tensor_2")
    self.assertEqual(subgraph.output[0].name, "tensor_3")
    self.assertEqual(subgraph.node[0].input[0], "tensor_2")
    self.assertEqual(subgraph.node[0].output[0], "tensor_3")


if __name__ == "__main__":
  absltest.main()
