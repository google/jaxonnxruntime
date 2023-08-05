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
import jax.numpy as jnp
from jaxonnxruntime.experimental import call_torch
import torch
from torch import nn
import torch.nn.functional as F


class SimpleModel(torch.nn.Module):

  def __init__(self):
    super(SimpleModel, self).__init__()
    self.fc1 = nn.Linear(3, 2)

  def forward(self, x):
    return F.relu(self.fc1(x))


class TestTorchToJax(absltest.TestCase):

  def test_torch_to_jax(self):
    # Input data
    input_data = torch.tensor([1.0, 2.0, 3.0])

    # Create and export the torch model
    torch_model = SimpleModel()
    torch_model.eval()
    jax_fn = call_torch.call_torch(torch_model, input_data)

    # Compare torch and jax output
    torch_output = torch_model(input_data).detach().numpy()
    jax_output = jax_fn(input_data.detach().numpy())[0]
    print(f"torch model result = {torch_output}")
    print(f"jaxonnxruntime model result = {jax_output}")
    self.assertTrue(jnp.allclose(jax_output, torch_output))


if __name__ == "__main__":
  absltest.main()
