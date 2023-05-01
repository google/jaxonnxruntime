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
"""Jaxonnxruntime python API example."""
from jaxonnxruntime import backend as jax_backend
import numpy as np
import torch

import onnx

MODEL_FILE = ".model.onnx"


def model():
  """A simple torch model to calculate addition of two tensors."""

  class Model(torch.nn.Module):

    def forward(self, x, y):
      return x.add(y)

  return Model()


def create_model(dtype: torch.dtype = torch.float32):
  """Create an instance of the model and export it to ONNX graph format, with dynamic size for the data."""
  sample_x = torch.ones(3, dtype=dtype)
  sample_y = torch.zeros(3, dtype=dtype)

  torch.onnx.export(
      model(),
      (sample_x, sample_y),
      MODEL_FILE,
      input_names=["x", "y"],
      output_names=["z"],
      dynamic_axes={"x": {0: "array_length_x"}, "y": {0: "array_length_y"}},
  )


def main():
  """main function."""
  create_model()
  onnx_model = onnx.load(MODEL_FILE)
  backend_rep = jax_backend.BackendRep(onnx_model)

  # Run the model on CPU consuming and producing numpy arrays
  def run(x: np.array, y: np.array) -> np.array:
    z = backend_rep.run({"x": x, "y": y})
    return z[0]

  print(run(x=np.float32([1.0, 2.0, 3.0]), y=np.float32([4.0, 5.0, 6.0])))
  # [array([5., 7., 9.], dtype=float32)]


if __name__ == "__main__":
  main()
