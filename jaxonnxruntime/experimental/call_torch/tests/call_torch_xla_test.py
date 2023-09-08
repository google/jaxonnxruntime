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

"""The call_torch_xla_p primitive test."""

from absl import logging
from absl.testing import absltest
import chex
import jax
from jaxonnxruntime.experimental import call_torch
from jaxonnxruntime.experimental.call_torch import call_torch_xla
import torch


def _convert_to_mhlo(jax_fn, inputs, *, dialect):
  lowered_forward = jax_fn.lower(*inputs)
  mhlo_text = lowered_forward.as_text(dialect=dialect)
  return mhlo_text


def _check_transforms(fn, inputs, *, dialect):
  jaxpr = jax.make_jaxpr(fn)(*inputs)
  logging.info(jaxpr)

  mhlo_text = jax.jit(fn).lower(*inputs).as_text(dialect=dialect)
  logging.info(mhlo_text)


class MhloTest(chex.TestCase):

  def _assert_all_close(self, expect_fn, actual_fn, inputs):
    expect_outputs = self.variant(expect_fn)(*inputs)
    actual_outputs = self.variant(actual_fn)(*inputs)
    logging.info(expect_outputs)
    logging.info(actual_outputs)
    chex.assert_trees_all_close(expect_outputs, actual_outputs)

  def test_basic(self):
    """Here we test basic pytorch function.

    We generate the stablehlo text offline for test purpose.
    """

    def foo(x, y):
      a = torch.sin(x)
      b = torch.cos(y)
      return a + b

    torch_inputs = (torch.randn(10, 10), torch.randn(10, 10))
    jax_inputs = jax.tree_map(call_torch.torch_tensor_to_np_array, torch_inputs)
    # This is generated from offline pytorch/xla
    foo_stablehlo_text = """
module @IrToHlo.13 attributes {mhlo.cross_program_prefetches = [], mhlo.dynamic_parameter_bindings = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<10x10xf32>, %arg1: tensor<10x10xf32>) -> tensor<10x10xf32> {
    %0 = stablehlo.sine %arg0 : tensor<10x10xf32>
    %1 = stablehlo.cosine %arg1 : tensor<10x10xf32>
    %2 = stablehlo.add %0, %1 : tensor<10x10xf32>
    return %2 : tensor<10x10xf32>
  }
}
"""

    mhlo_module = call_torch_xla.MhloModule(
        module=foo_stablehlo_text, fun_name=foo.__name__
    )

    chex.assert_trees_all_close(
        call_torch_xla.call_torch_xla(*jax_inputs, module=mhlo_module),
        foo(*torch_inputs),
        rtol=1e-6,
        atol=1e-5,
    )


if __name__ == "__main__":
  absltest.main()
