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
from jaxonnxruntime.onnx_ops import dropout
import numpy as np
import onnx

NodeProto = onnx.NodeProto
AttributeProto = onnx.AttributeProto
Dropout = dropout.Dropout
OnnxNode = onnx_node.OnnxNode


class DropoutTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    node_proto = NodeProto(
        op_type='Dropout', input=['input'], output=['output']
    )
    attr = AttributeProto()
    attr.name = 'seed'
    attr.i = int(0)
    attr.type = AttributeProto.INT
    node_proto.attribute.extend([attr])
    self.node_dropout = OnnxNode(node_proto)

  def test_dropout_ratio_trainmode(self):
    data = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32
    )
    ratio = np.array([0.0], dtype=np.float32)
    training_mode = True
    inputs = [data, ratio, training_mode]

    dropout_func = Dropout.version_13(self.node_dropout, inputs)

    outputs = dropout_func(*inputs, **self.node_dropout.attrs_dict)

    expect = data.copy()
    np.testing.assert_array_equal(outputs, expect)

  def test_dropout_static_ratio(self):
    data = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32
    )
    ratio = np.array([0.0], dtype=np.float32)
    training_mode = True
    inputs = [data, ratio, training_mode]

    dropout_func = Dropout.version_13(self.node_dropout, inputs)

    outputs_ratio_0 = dropout_func(*inputs, **self.node_dropout.attrs_dict)

    ratio = np.array([1.0], dtype=np.float32)
    inputs = [data, ratio, training_mode]
    outputs_ratio_1 = dropout_func(*inputs, **self.node_dropout.attrs_dict)

    np.testing.assert_array_equal(outputs_ratio_0, outputs_ratio_1)

    expect_outputs_ratio_1 = np.zeros_like(data)
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        outputs_ratio_1,
        expect_outputs_ratio_1,
    )

  def test_dropout_stats_trainmode(self):
    data = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32
    )
    ratio = np.array([1.0], dtype=np.float32)
    training_mode = True
    inputs = [data, ratio, training_mode]

    dropout_func = Dropout.version_13(self.node_dropout, inputs)

    outputs_train = dropout_func(*inputs, **self.node_dropout.attrs_dict)

    training_mode = False
    inputs = [data, ratio, training_mode]
    outputs_infer = dropout_func(*inputs, **self.node_dropout.attrs_dict)

    np.testing.assert_array_equal(outputs_train, outputs_infer)

    expect_outputs_infer = data.copy()
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        outputs_infer,
        expect_outputs_infer,
    )


if __name__ == '__main__':
  absltest.main()
