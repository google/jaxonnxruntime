# Copyright 2024 The Jaxonnxruntime Authors.
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

import abc

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
from jax.experimental import jax2tf
from jaxonnxruntime.experimental import call_torch
from jaxonnxruntime.experimental.call_torch import call_torch_xla
import tensorflow as tf
import torch


class TorchModuleBase(abc.ABC):

  @abc.abstractmethod
  def torch_func(self):
    pass

  @abc.abstractmethod
  def torch_inputs(self):
    pass

  def torch_params(self):
    pass

  @abc.abstractmethod
  def stablehlo_text(self):
    pass


class TorchModuleFoo(TorchModuleBase):

  def torch_func(self):
    def foo(params, inputs):  # pylint: disable=unused-argument
      x, y = inputs
      a = torch.sin(x)
      b = torch.cos(y)
      return a + b

    return foo

  def torch_params(self):
    return ()

  def torch_inputs(self):
    return (torch.randn(10, 10), torch.randn(10, 10))

  def stablehlo_text(self):
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
    return foo_stablehlo_text


class CallTorchXlaTest(chex.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="foo",
          torch_module=TorchModuleFoo(),
      ),
  )
  def test_basic(self, torch_module: TorchModuleBase):
    """Here we test basic pytorch function."""
    torch_func = torch_module.torch_func()
    torch_inputs = torch_module.torch_inputs()
    torch_params = torch_module.torch_params()
    torch_results = torch_func(torch_params, torch_inputs)
    _, res_tree_def = jax.tree_flatten(
        jax.tree_map(call_torch.torch_tensor_to_jax_array, torch_results)
    )
    print(f"res_tree_def = {res_tree_def}")
    stablehlo_text = torch_module.stablehlo_text()
    jax_params = jax.tree_map(
        call_torch.torch_tensor_to_jax_array, torch_params
    )
    jax_inputs = jax.tree_map(
        call_torch.torch_tensor_to_jax_array, torch_inputs
    )

    def jax_func(jax_params, jax_inputs):
      flat_xla_args = jax.tree_leaves(jax_params) + jax.tree_leaves(jax_inputs)
      flat_xla_res = call_torch_xla.call_torch_xla(
          *flat_xla_args, module=stablehlo_text
      )
      return jax.tree_unflatten(res_tree_def, flat_xla_res)

    chex.assert_trees_all_close(
        jax_func(jax_params, jax_inputs),
        torch_results,
        rtol=1e-6,
        atol=1e-5,
    )

    tf_func = tf.function(
        jax2tf.convert(jax.jit(jax_func), with_gradient=False, enable_xla=True),
        jit_compile=True,
        autograph=False,
    )

    chex.assert_trees_all_close(
        tf_func(jax_params, jax_inputs), torch_results, atol=1e-5, rtol=1e-6
    )

if __name__ == "__main__":
  absltest.main()
