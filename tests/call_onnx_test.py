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
import jax
from jaxonnxruntime import call_onnx
from jaxonnxruntime import config_class
import numpy as np

import onnx


def create_test_model(x: np.ndarray) -> onnx.ModelProto:
  input_tensor = onnx.ValueInfoProto(
      name='input',
      type=onnx.TypeProto(
          tensor_type=onnx.TypeProto.Tensor(
              elem_type=onnx.TensorProto.FLOAT,
              shape=onnx.TensorShapeProto(
                  dim=[
                      onnx.TensorShapeProto.Dimension(dim_value=d)
                      for d in x.shape
                  ]
              ),
          )
      ),
  )
  output_tensor = onnx.ValueInfoProto(
      name='output',
      type=onnx.TypeProto(
          tensor_type=onnx.TypeProto.Tensor(
              elem_type=onnx.TensorProto.FLOAT,
              shape=onnx.TensorShapeProto(
                  dim=[
                      onnx.TensorShapeProto.Dimension(dim_value=d)
                      for d in x.shape
                  ]
              ),
          )
      ),
  )
  node_abs = onnx.NodeProto(op_type='Abs', input=['input'], output=['output'])
  graph_def = onnx.GraphProto(
      node=[node_abs],
      name='abs_graph',
      input=[input_tensor],
      output=[output_tensor],
  )
  model_proto = onnx.ModelProto(graph=graph_def, producer_name='onnx-example')
  return model_proto


class TestCallOnnx(absltest.TestCase):

  def test_basic(self):
    x = np.array([-2.0, 1.0, 3.0], dtype=np.float32)
    model_proto = create_test_model(x)
    jax_func, model_params = call_onnx.call_onnx_model(model_proto, [x])
    results = jax_func(model_params, [x])
    expect = [np.array([2.0, 1.0, 3.0], dtype=np.float32)]
    np.testing.assert_array_equal(results, expect)

    with config_class.jaxort_experimental_support_abtract_input_shape(True):
      x = np.array([-2.0, -8.0, 3.0], dtype=np.float32)
      jax_func, model_params = call_onnx.call_onnx_model(
          model_proto, [jax.ShapeDtypeStruct(x.shape, x.dtype)]
      )
      results = jax_func(model_params, [x])
      expect = [np.array([2.0, 8.0, 3.0], dtype=np.float32)]
      np.testing.assert_array_equal(results, expect)


if __name__ == '__main__':
  absltest.main()
