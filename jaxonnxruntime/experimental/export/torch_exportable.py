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

"""Torch Exportable class."""

import dataclasses
from typing import Any
from absl import logging
import jax
from jaxonnxruntime.experimental import call_torch
from jaxonnxruntime.experimental.export import exportable
from jaxonnxruntime.experimental.export import exportable_utils
import torch

Tensor = torch.Tensor
Sharding = exportable_utils.Sharding
HloSharding = exportable_utils.HloSharding


@dataclasses.dataclass
class TensorWithSharding:
  tensor: Tensor
  sharding: Sharding = None


@dataclasses.dataclass
class TorchExportable(exportable.Exportable):
  """Torch Exportable class."""

  _out_hlo_sharding: tuple[HloSharding, ...] | None = None

  @classmethod
  def _to_xla_hlo_sharding(
      cls,
      sharding: jax.sharding.Sharding,
      tensor: Tensor,
  ) -> HloSharding:
    if sharding is None:
      return None
    return sharding._to_xla_hlo_sharding(tensor.ndim)  # pylint: disable=protected-access

  def __post_init__(self):
    assert not self.kwargs, "kwargs is not supported currently."
    self.args_maybe_sharding_flat, self._in_tree = jax.tree_util.tree_flatten(
        (self.args, self.kwargs)
    )

    self.jit_func = torch.jit.trace(
        self.function,
        example_inputs=self.args,
        # example_kwarg_inputs=self.kwargs,
    )
    res = self.jit_func(
        *self.args,
        # **self.kwargs
    )
    res_flat, self._out_tree = jax.tree_util.tree_flatten(res)
    out_avals = jax.tree_util.tree_map(
        exportable_utils.torch_tensor_to_jax_abstract_shaped_array,
        res_flat,
    )
    self._out_avals = tuple(out_avals)

  @property
  def in_tree(self):
    return self._in_tree

  @property
  def out_tree(self):
    return self._out_tree

  @property
  def fun_name(self) -> str:
    return getattr(self.function, "__name__", "unknown")

  @property
  def in_avals(self) -> tuple[jax.core.AbstractValue, ...]:
    def _to_jax_abstract_shaped_array(x):
      if isinstance(x, TensorWithSharding):
        x = x.tensor
      return exportable_utils.torch_tensor_to_jax_abstract_shaped_array(x)

    args_avals_flat = jax.tree_util.tree_map(
        _to_jax_abstract_shaped_array,
        self.args_maybe_sharding_flat,
    )
    return tuple(args_avals_flat)

  @property
  def module_kept_var_idx(self) -> tuple[int, ...]:
    return tuple(range(len(self.in_avals)))

  @property
  def in_shardings_hlo(self) -> tuple[HloSharding, ...]:

    hlo_in_shardings = jax.tree_util.tree_map(
        lambda x: self._to_xla_hlo_sharding(x.sharding, x.tensor)
        if isinstance(x, TensorWithSharding)
        else None,
        self.args_maybe_sharding_flat,
    )
    return tuple(hlo_in_shardings)

  @property
  def out_shardings_hlo(self) -> tuple[HloSharding, ...]:
    # Do we need really provide the out_sharding?
    if self._out_hlo_sharding is None:
      return jax.tree_util.tree_map(lambda x: None, self.out_avals)
    return tuple(self._out_hlo_sharding)

  @property
  def out_avals(self):
    return self._out_avals

  @out_shardings_hlo.setter
  def out_shardings(self, jax_shardings: tuple[Sharding, ...]):
    """Provide the function to change out_sharding."""
    out_aval = self.out_avals
    hlo_sharding = jax.tree_util.tree_map(
        self._to_xla_hlo_sharding,
        jax_shardings,
        out_aval,
    )
    self._out_hlo_sharding = hlo_sharding

  @property
  def nr_devices(self) -> int:
    return len(jax.devices())

  @property
  def mlir_module(self) -> Any:
    """Returns the mlir module from torch."""
    args_flat = jax.tree_util.tree_map(
        lambda x: x.tensor if isinstance(x, TensorWithSharding) else x,
        self.args_maybe_sharding_flat,
    )
    jax_fn, jax_params = call_torch.call_torch(self.jit_func, self.args)
    jax_fn = jax.jit(jax_fn)
    args_jax_flat = jax.tree_util.tree_map(jax.dlpack.from_dlpack, args_flat)
    logging.info("args_jax_flat: %s", args_jax_flat)
    lowered = jax_fn.lower(jax_params, args_jax_flat)
    lowering = lowered._lowering  # pylint: disable=protected-access
    mlir_module = lowering.stablehlo()
    logging.info("mlir_module: %s", mlir_module)
    return mlir_module

  @property
  def mlir_module_serialized(self) -> bytes:
    return exportable_utils.serialize_stablehlo_mlir_module(self.mlir_module)

  @property
  def ordered_effects(self):
    return tuple()

  @property
  def unordered_effects(self):
    return tuple()
