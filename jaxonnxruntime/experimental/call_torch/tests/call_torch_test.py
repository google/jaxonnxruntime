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

import unittest

from absl.testing import absltest
import jax
import jax.numpy as jnp
from jaxonnxruntime.experimental import call_torch
from jaxonnxruntime.experimental.call_torch.tests.d2l_torch import d2l
import torch
from torch import nn
import torch.nn.functional as F


class TestCallTorchBasic(call_torch.CallTorchTestCase):

  def test_torch_module(self):
    class SimpleModel(torch.nn.Module):

      def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(3, 2)

      def forward(self, x):
        return F.relu(self.fc1(x))

    torch_inputs = (torch.tensor([1.0, 2.0, 3.0]),)
    test_module = SimpleModel()
    self.assert_call_torch_convert_and_compare(test_module, torch_inputs)

  def test_torch_function(self):
    def torch_func(x, y):
      return 2 * x + y

    torch_inputs = (torch.rand(3), torch.rand(3))
    self.assert_call_torch_convert_and_compare(torch_func, torch_inputs)


class TestCh11AttentionTransformer(call_torch.CallTorchTestCase):

  @unittest.skip(
      "torch.jit.script fail here. Need report PyTorch RD and solve it on"
      " pytorch codebase."
  )
  def test_multi_head_attention(self):
    num_hiddens, num_heads = 100, 5
    attention = d2l.MultiHeadAttention(num_hiddens, num_heads, 0.5)
    batch_size, num_queries, num_kvpairs = 2, 4, 6
    valid_lens = torch.tensor([3, 2])
    x = torch.ones((batch_size, num_queries, num_hiddens))
    y = torch.ones((batch_size, num_kvpairs, num_hiddens))
    torch_inputs = (x, y, y, valid_lens)
    torch_outputs = attention(*torch_inputs)
    self.assertEqual(
        torch_outputs.shape, (batch_size, num_queries, num_hiddens)
    )
    jax_inputs = jax.tree_map(call_torch.torch_tensor_to_np_array, torch_inputs)
    jax_fn, jax_params = call_torch.call_torch(attention, torch_inputs)
    jax_outputs = jax_fn(jax_params, jax_inputs)
    self.assertTrue(jnp.allclose(jax_outputs, torch_outputs))


if __name__ == "__main__":
  absltest.main()
