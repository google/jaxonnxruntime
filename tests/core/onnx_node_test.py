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
from jaxonnxruntime.core import onnx_node

import onnx


OnnxNode = onnx_node.OnnxNode
convert_onnx = onnx_node.convert_onnx


class TestOnnxNode(absltest.TestCase):

  def test_onnx_node_init(self):
    # Create a dummy NodeProto object
    node_proto = onnx.NodeProto()
    node_proto.name = "test_node"
    node_proto.op_type = "Add"
    node_proto.domain = "test_domain"
    node_proto.attribute.add(name="test_attr", i=42)
    node_proto.input.extend(["input1", "input2"])
    node_proto.output.extend(["output1", "output2"])

    # Create an OnnxNode object
    node = OnnxNode(node_proto)

    # Test that the attributes were correctly set
    self.assertEqual(node.name, "test_node")
    self.assertEqual(node.op_type, "Add")
    self.assertEqual(node.domain, "test_domain")
    self.assertEqual(node.attrs["test_attr"], 42)
    self.assertEqual(node.inputs, ["input1", "input2"])
    self.assertEqual(node.outputs, ["output1", "output2"])
    self.assertEqual(node.node_proto, node_proto)
    self.assertIsNone(node.context_graph)

  def test_convert_onnx(self):
    # Test converting a few different types of attributes
    attr_proto = onnx.AttributeProto()
    attr_proto.f = 3.14
    self.assertLess(abs(float(convert_onnx(attr_proto)) - 3.14), 0.001)

    attr_proto = onnx.AttributeProto()
    attr_proto.i = 42
    self.assertEqual(convert_onnx(attr_proto), 42)

    attr_proto = onnx.AttributeProto()
    attr_proto.s = b"test_string"
    self.assertEqual(convert_onnx(attr_proto), "test_string")

    attr_proto = onnx.AttributeProto()
    tensor_proto = onnx.TensorProto()
    tensor_proto.dims.extend([2, 3])
    tensor_proto.float_data.extend([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    attr_proto.t.CopyFrom(tensor_proto)
    self.assertEqual(convert_onnx(attr_proto), tensor_proto)


if __name__ == "__main__":
  absltest.main()
