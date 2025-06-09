# Copyright 2025 The Jaxonnxruntime Authors.
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

import os
from absl import logging
from absl.testing import absltest
from jaxonnxruntime.core import config_class
from jaxonnxruntime.experimental import call_torch
from jaxonnxruntime.experimental.call_torch.test_data.d2l_torch import d2l
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
    output_dir = os.getenv(
        "TEST_UNDECLARED_OUTPUTS_DIR", "/tmp/torch_repeat_interleave"
    )
    torch_inputs = (torch.tensor([[1, 2], [3, 4]]), torch.tensor(2))
    torch_func = torch.repeat_interleave
    with config_class.jaxort_only_allow_initializers_as_static_args(False):
      self.assert_call_torch_convert_and_compare(
          torch_func, torch_inputs, onnx_dump_prefix=output_dir
      )


class TestCh11AttentionTransformer(call_torch.CallTorchTestCase):
  # https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html

  def test_masked_softmax(self):
    torch_inputs = (torch.rand(2, 2, 4), torch.tensor([2, 3]))
    torch_func = d2l.masked_softmax
    with config_class.jaxort_only_allow_initializers_as_static_args(False):
      self.assert_call_torch_convert_and_compare(torch_func, torch_inputs)

  def test_additive_attention(self):
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

  def test_multi_head_attention(self):
    num_hiddens, num_heads = 100, 5
    batch_size, num_queries, num_kvpairs = 2, 4, 6
    valid_lens = torch.tensor([3, 2])
    valid_lens = torch.repeat_interleave(valid_lens, repeats=num_heads, dim=0)
    x = torch.normal(0, 1, (batch_size, num_queries, num_hiddens))
    y = torch.normal(0, 1, (batch_size, num_kvpairs, num_hiddens))
    torch_inputs = (x, y, y, valid_lens)
    torch_func = d2l.MultiHeadAttention(num_hiddens, num_heads, 0.5)
    torch_func.eval()
    with config_class.jaxort_only_allow_initializers_as_static_args(False):
      self.assert_call_torch_convert_and_compare(torch_func, torch_inputs)

  def test_torch_loop(self):
    torch_inputs = (torch.randn(128),)

    def f(x):
      for _ in range(100):
        x = torch.sin(x)
      return x

    torch_func = f
    logging.info(torch_func)
    self.assert_call_torch_convert_and_compare(torch_func, torch_inputs)

  def test_torch_bfloat16(self):
    # Create sample tensors in bfloat16 format
    torch_inputs = (
        torch.randn(4, 4, dtype=torch.bfloat16),
        torch.randn(4, 4, dtype=torch.bfloat16),
    )
    torch_func = torch.matmul
    self.assert_call_torch_convert_and_compare(
        torch_func, torch_inputs, verbose=True
    )


class TestCallTorchNN(call_torch.CallTorchTestCase):
  """test torch.nn layers https://pytorch.org/docs/stable/nn.html."""

  def test_conv1d(self):
    torch_func = torch.nn.Conv1d(16, 33, 3, stride=2)
    torch_inputs = (torch.randn(20, 16, 50),)
    self.assert_call_torch_convert_and_compare(torch_func, torch_inputs)

  def test_conv2d(self):
    torch_func = nn.Conv2d(
        16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1)
    )
    x = torch.randn(20, 16, 50, 100)
    torch_inputs = (x,)
    self.assert_call_torch_convert_and_compare(torch_func, torch_inputs)

  def test_conv3d(self):
    torch_func = nn.Conv3d(
        16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0)
    )
    x = torch.randn(20, 16, 10, 50, 100)
    torch_inputs = (x,)
    # TODO(johnqiangzhang): investigate why relative error is big.
    # Max absolute difference: 3.33786e-06
    # Max relative difference: 10.1
    self.assert_call_torch_convert_and_compare(
        torch_func, torch_inputs, rtol=10.2
    )

  @absltest.skip("NotImplementedError: ConvTranspose is not implemented.")
  def test_conv_transpose2d(self):
    torch_func = torch.nn.ConvTranspose2d(
        16, 33, (3, 5), stride=(2, 1), padding=(4, 2)
    )
    torch_inputs = (torch.randn(20, 16, 50, 100),)
    self.assert_call_torch_convert_and_compare(torch_func, torch_inputs)

    def f(x):
      downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
      upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
      h = downsample(x)
      logging.info(h.size())
      output = upsample(h, output_size=h.size())
      logging.info(output.size())
      return output

    torch_func = f
    torch_inputs = (torch.randn(1, 16, 12, 12),)
    self.assert_call_torch_convert_and_compare(torch_func, torch_inputs)

  @absltest.skip("NotImplementedError: ConvTranspose is not implemented.")
  def test_conv_transpose3d(self):
    torch_func = nn.ConvTranspose3d(
        16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(0, 4, 2)
    )
    torch_inputs = (torch.randn(20, 16, 10, 50, 100),)
    self.assert_call_torch_convert_and_compare(torch_func, torch_inputs)

  def test_max_pool1d(self):
    torch_func = nn.MaxPool1d(3, stride=2)
    torch_inputs = (torch.randn(20, 16, 50),)
    self.assert_call_torch_convert_and_compare(torch_func, torch_inputs)

  def test_max_pool2d(self):
    torch_func = nn.MaxPool2d(3, stride=(2, 1))
    torch_inputs = (torch.randn(20, 16, 50, 32),)
    self.assert_call_torch_convert_and_compare(torch_func, torch_inputs)

  def test_max_pool3d(self):
    torch_func = nn.MaxPool3d(3, stride=2)
    torch_inputs = (torch.randn(20, 16, 50, 44, 31),)
    self.assert_call_torch_convert_and_compare(torch_func, torch_inputs)

  @absltest.skip(
      "torch.onnx.errors.UnsupportedOperatorError: Exporting the operator"
      " 'aten::max_unpool2d' to ONNX opset version 17 is not supported."
  )
  def test_max_unpool1d(self):
    def f(x):
      pool = nn.MaxPool1d(2, stride=2, return_indices=True)
      unpool = nn.MaxUnpool1d(2, stride=2)
      y, indices = pool(x)
      return unpool(y, indices)

    torch_func = f
    torch_inputs = (torch.tensor([[[1.0, 2, 3, 4, 5, 6, 7, 8]]]),)
    self.assert_call_torch_convert_and_compare(torch_func, torch_inputs)

  def test_avg_pool1d(self):
    torch_func = nn.AvgPool1d(3, stride=2)
    torch_inputs = (torch.tensor([[[1.0, 2, 3, 4, 5, 6, 7]]]),)
    self.assert_call_torch_convert_and_compare(torch_func, torch_inputs)

  def test_avg_pool2d(self):
    torch_func = nn.AvgPool2d((3, 2), stride=(2, 1))
    torch_inputs = (torch.randn(20, 16, 50, 32),)
    self.assert_call_torch_convert_and_compare(torch_func, torch_inputs)

  def test_avg_pool3d(self):
    torch_func = nn.AvgPool3d(3, stride=(2, 1, 2))
    torch_inputs = (torch.randn(20, 16, 50, 44, 31),)
    self.assert_call_torch_convert_and_compare(torch_func, torch_inputs)

  @absltest.skip("NotImplementedError: Sign is not implemented.")
  def test_lp_pool1d(self):
    torch_func = nn.LPPool1d(2, 3, stride=2)
    torch_inputs = (torch.randn(20, 16, 50),)
    self.assert_call_torch_convert_and_compare(torch_func, torch_inputs)

  @absltest.skip("NotImplementedError: Sign is not implemented.")
  def test_lp_pool2d(self):
    torch_func = nn.LPPool2d(2, 3, stride=2)
    torch_inputs = (torch.randn(20, 16, 50, 32),)
    self.assert_call_torch_convert_and_compare(torch_func, torch_inputs)


if __name__ == "__main__":
  absltest.main()
