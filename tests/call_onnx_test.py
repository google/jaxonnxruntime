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
from jaxonnxruntime import call_onnx
import numpy as np

import onnx

ValueInfoProto = onnx.ValueInfoProto
TypeProto = onnx.TypeProto
GraphProto = onnx.GraphProto
NodeProto = onnx.NodeProto
ModelProto = onnx.ModelProto
TensorProto = onnx.TensorProto
TensorShapeProto = onnx.TensorShapeProto


class TestCallOnnx(absltest.TestCase):

  def test_basic(self):
    x = np.array([-2.0, 1.0, 3.0], dtype=np.float32)

    input_tensor = ValueInfoProto(
        name='input',
        type=TypeProto(
            tensor_type=TypeProto.Tensor(
                elem_type=TensorProto.FLOAT,
                shape=TensorShapeProto(
                    dim=[
                        TensorShapeProto.Dimension(dim_value=d) for d in x.shape
                    ]
                ),
            )
        ),
    )
    output_tensor = ValueInfoProto(
        name='output',
        type=TypeProto(
            tensor_type=TypeProto.Tensor(
                elem_type=TensorProto.FLOAT,
                shape=TensorShapeProto(
                    dim=[
                        TensorShapeProto.Dimension(dim_value=d) for d in x.shape
                    ]
                ),
            )
        ),
    )
    node_abs = NodeProto(op_type='Abs', input=['input'], output=['output'])
    graph_def = GraphProto(
        node=[node_abs],
        name='abs_graph',
        input=[input_tensor],
        output=[output_tensor],
    )
    model_proto = ModelProto(graph=graph_def, producer_name='onnx-example')
    jax_func, model_params = call_onnx.call_onnx(model_proto, [x])
    results = jax_func(model_params, [x])
    expect = [np.array([2.0, 1.0, 3.0], dtype=np.float32)]
    np.testing.assert_array_equal(results, expect)


if __name__ == '__main__':
  absltest.main()
