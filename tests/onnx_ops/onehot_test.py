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

from absl.testing import absltest
from jaxonnxruntime.core import onnx_node
from jaxonnxruntime.onnx_ops import onehot
import numpy as np

import onnx

NodeProto = onnx.NodeProto
AttributeProto = onnx.AttributeProto
OneHot = onehot.OneHot
OnnxNode = onnx_node.OnnxNode


class OneHotTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    node_proto = NodeProto(op_type='OneHot', input=['input'], output=['output'])
    self.node_onehot = OnnxNode(node_proto)

  def test_onehot(self):
    indices = np.array([0, -7, -8], dtype=np.int64)
    depth = np.float32(10)
    off_value, on_value = 1, 3
    values = np.array([off_value, on_value], dtype=np.float32)
    inputs = [indices, depth, values]

    onehot_func = OneHot.version_11(self.node_onehot, inputs)

    outputs = onehot_func(*inputs, **self.node_onehot.attrs_dict)

    expect = np.array([
        [3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ])
    np.testing.assert_array_equal(outputs, expect)

  def test_onehot_static_depth(self):
    indices = np.array([0, -7, -8], dtype=np.int64)
    depth = np.float32(10)
    off_value, on_value = 1, 3
    values = np.array([off_value, on_value], dtype=np.float32)
    inputs = [indices, depth, values]

    onehot_func = OneHot.version_11(self.node_onehot, inputs)

    outputs_depth_10 = onehot_func(*inputs, **self.node_onehot.attrs_dict)

    depth = np.float32(8)
    inputs = [indices, depth, values]
    outputs_depth_8 = onehot_func(*inputs, **self.node_onehot.attrs_dict)

    np.testing.assert_array_equal(outputs_depth_10, outputs_depth_8)

    expect_outputs_depth_8 = np.array([
        [3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ])
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        outputs_depth_8,
        expect_outputs_depth_8,
    )


if __name__ == '__main__':
  absltest.main()
