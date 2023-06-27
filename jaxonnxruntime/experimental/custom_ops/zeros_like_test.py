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
from jaxonnxruntime import call_onnx
from jaxonnxruntime.experimental import custom_ops  # pylint: disable=unused-import
import numpy as np

import onnx


class ZerosLikeTest(absltest.TestCase):

  def test_basic(self):
    # Create input and output tensors.
    input_tensor = onnx.helper.make_tensor_value_info(
        "X", onnx.TensorProto.FLOAT, [1, 3, 224, 224]
    )
    output_tensor = onnx.helper.make_tensor_value_info(
        "Y", onnx.TensorProto.FLOAT, [1, 3, 224, 224]
    )

    # Create the ZerosLike node
    node = onnx.helper.make_node(
        "ZerosLike",
        inputs=["X"],
        outputs=["Y"],
        dtype=onnx.TensorProto.FLOAT,
        domain="jaxonnxruntime",
    )

    # Create the graph with the node
    graph_def = onnx.helper.make_graph(
        [node],
        "ZerosLike_Model",
        [input_tensor],
        [output_tensor],
    )

    # Create the model
    onnx_model = onnx.helper.make_model(
        graph_def,
        producer_name="JAX-ONNX",
        opset_imports=[
            onnx.helper.make_opsetid(
                onnx.defs.ONNX_DOMAIN, onnx.defs.onnx_opset_version()
            ),
            onnx.helper.make_opsetid("jaxonnxruntime", 1),
        ],
    )
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.zeros_like(x).astype(np.float32)
    inputs = [x]
    jax_model_func, jax_model_params = call_onnx.call_onnx_model(
        onnx_model, inputs
    )
    outputs = jax_model_func(jax_model_params, inputs)
    expect_outputs = [y]
    self.assertLen(outputs, 1, f"output is {outputs}")
    np.testing.assert_allclose(outputs[0], expect_outputs[0], atol=1e-6)


if __name__ == "__main__":
  absltest.main()
