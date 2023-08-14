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
from jaxonnxruntime import config_class
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

  def test_torch_repeat_interleave(self):
    torch_inputs = (torch.tensor([[1, 2], [3, 4]]), torch.tensor(2))
    torch_func = torch.repeat_interleave
    with config_class.jaxort_only_allow_initializers_as_static_args(False):
      self.assert_call_torch_convert_and_compare(torch_func, torch_inputs)


class TestCh11AttentionTransformer(call_torch.CallTorchTestCase):
  # https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html

  def test_masked_softmax(self):
    torch_inputs = (torch.rand(2, 2, 4), torch.tensor([2, 3]))
    torch_func = d2l.masked_softmax
    with config_class.jaxort_only_allow_initializers_as_static_args(False):
      self.assert_call_torch_convert_and_compare(torch_func, torch_inputs)

  def test_addictive_attention(self):
    queries = torch.normal(0, 1, (2, 1, 20))
    keys = torch.normal(0, 1, (2, 10, 2))
    values = torch.normal(0, 1, (2, 10, 4))
    valid_lens = torch.tensor([2, 6])
    torch_inputs = (queries, keys, values, valid_lens)
    attention = d2l.AdditiveAttention(num_hiddens=8, dropout=0.1)
    torch_func = attention.eval()
    with config_class.jaxort_only_allow_initializers_as_static_args(False):
      self.assert_call_torch_convert_and_compare(torch_func, torch_inputs)

  def test_bmm(self):
    torch_inputs = (torch.ones((2, 3, 4)), torch.ones((2, 4, 6)))
    torch_func = torch.bmm
    self.assert_call_torch_convert_and_compare(torch_func, torch_inputs)

  def test_dot_product_attention(self):
    queries = torch.normal(0, 1, (2, 1, 2))
    keys = torch.normal(0, 1, (2, 10, 2))
    values = torch.normal(0, 1, (2, 10, 4))
    valid_lens = torch.tensor([2, 6])
    torch_inputs = (queries, keys, values, valid_lens)
    torch_func = d2l.DotProductAttention(dropout=0.5).eval()
    with config_class.jaxort_only_allow_initializers_as_static_args(False):
      self.assert_call_torch_convert_and_compare(torch_func, torch_inputs)

  @unittest.skip("TorchONNXExportError")
  def test_multi_head_attention(self):
    num_hiddens, num_heads = 100, 5
    batch_size, num_queries, num_kvpairs = 2, 4, 6
    valid_lens = torch.tensor([3, 2])
    x = torch.ones((batch_size, num_queries, num_hiddens))
    y = torch.ones((batch_size, num_kvpairs, num_hiddens))
    torch_inputs = (x, y, y, valid_lens)
    torch_func = d2l.MultiHeadAttention(num_hiddens, num_heads, 0.5)
    self.assert_call_torch_convert_and_compare(torch_func, torch_inputs)


if __name__ == "__main__":
  absltest.main()
