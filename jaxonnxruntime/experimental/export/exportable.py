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

"""Exported base class."""

from collections.abc import Callable, Sequence
import dataclasses
from typing import Any

import jax
from jax.experimental import export as jax_export
from jax.interpreters import mlir
from jaxonnxruntime.experimental.export import exportable_utils


HloSharding = exportable_utils.HloSharding


@dataclasses.dataclass
class Exportable:
  """Exportable base class."""

  function: Callable[..., Any]
  args: tuple[Any, ...]
  kwargs: dict[str, Any]
  lowering_platforms: Sequence[str] | None = None

  @classmethod
  def _to_xla_hlo_sharding(
      cls, s: Any, aval: jax.core.ShapedArray
  ) -> HloSharding:
    if str(s) == "UnspecifiedValue":
      return None
    return s._to_xla_hlo_sharding(aval.ndim)  # pylint: disable=protected-access

  def __post_init__(self):
    if not hasattr(self.function, "lower"):
      wrapped_func_jax = jax.jit(self.function)
    else:
      wrapped_func_jax = self.function

    self.lowered = wrapped_func_jax.lower(
        *self.args,
        **self.kwargs,
        _experimental_lowering_parameters=mlir.LoweringParameters(
            platforms=self.actual_lowering_platforms,
        ),
    )
    self.lowering = self.lowered._lowering  # pylint: disable=protected-access

  @property
  def actual_lowering_platforms(self) -> tuple[str, ...]:
    if self.lowering_platforms is not None:
      return tuple(self.lowering_platforms)
    else:
      return (jax_export.default_lowering_platform(),)

  @property
  def fun_name(self) -> str:
    return getattr(self.function, "__name__", "unknown")

  @property
  def in_avals(self):
    args_avals_flat, _ = jax.tree_util.tree_flatten(self.lowered.in_avals)
    return tuple(args_avals_flat)

  @property
  def out_avals(self):
    lowering = self.lowering
    # Figure out the result types and shapes
    if "global_out_avals" in lowering.compile_args:
      # This is currently the case for pjit
      out_avals_flat = lowering.compile_args["global_out_avals"]
    elif "shards" in lowering.compile_args:  # for PmapComputation
      out_avals_flat = lowering.compile_args["shards"].out_sharded_avals
    else:
      out_avals_flat = self.lowered.compile_args["out_avals"]
    return tuple(out_avals_flat)

  @property
  def module_kept_var_idx(self):
    lowering = self.lowering
    if "kept_var_idx" in lowering.compile_args:
      module_kept_var_idx = tuple(sorted(lowering.compile_args["kept_var_idx"]))
    else:
      # For pmap
      module_kept_var_idx = tuple(range(len(self.in_avals)))
    return module_kept_var_idx

  @property
  def in_shardings(self) -> tuple[HloSharding, ...]:
    lowering = self.lowering
    module_kept_var_idx = self.module_kept_var_idx
    in_shardings = lowering.compile_args["in_shardings"]
    assert len(in_shardings) == len(self.module_kept_var_idx)
    all_in_shardings = [None] * len(module_kept_var_idx)
    for idx, in_s in zip(sorted(module_kept_var_idx), in_shardings):
      all_in_shardings[idx] = in_s

    return tuple(
        self._to_xla_hlo_sharding(s, aval)
        for s, aval in zip(all_in_shardings, self.in_avals)
    )

  @property
  def out_shardings(self) -> tuple[HloSharding, ...]:
    return tuple(
        self._to_xla_hlo_sharding(s, aval)
        for s, aval in zip(
            self.lowering.compile_args["out_shardings"], self.out_avals
        )
    )

  @property
  def nr_devices(self) -> int:
    lowering = self.lowering
    nr_devices = len(lowering.compile_args["device_assignment"])
    return nr_devices

  @property
  def mlir_module_serialized(self) -> bytes:
    mlir_module = self.lowering.stablehlo()
    return exportable_utils.serialize_stablehlo_mlir_module(mlir_module)

  @property
  def mlir_module_serialization_version(self) -> int:
    """Returns the jax_serialization_version of the mlir module."""
    version = jax.config.jax_serialization_version
    minimum_supported_serialization_version = (
        jax_export.minimum_supported_serialization_version
    )
    maximum_supported_serialization_version = (
        jax_export.maximum_supported_serialization_version
    )
    if (
        version < minimum_supported_serialization_version
        or version > maximum_supported_serialization_version
    ):
      raise ValueError(
          f"The requested jax_serialization version {version} is outside the "
          "range of supported versions"
          f" [{minimum_supported_serialization_version}"
          f"..{maximum_supported_serialization_version}]"
      )
    return version

  def export(self) -> jax_export.Exported:
    return jax_export.Exported(
        fun_name=self.fun_name,
        in_tree=self.lowered.in_tree,
        out_tree=self.lowered.out_tree,
        in_avals=self.in_avals,
        out_avals=self.out_avals,
        in_shardings=self.in_shardings,
        out_shardings=self.out_shardings,
        nr_devices=self.nr_devices,
        lowering_platforms=self.actual_lowering_platforms,
        ordered_effects=tuple(),
        unordered_effects=tuple(),
        disabled_safety_checks=tuple(),
        mlir_module_serialized=self.mlir_module_serialized,
        module_kept_var_idx=self.module_kept_var_idx,
        uses_shape_polymorphism=False,
        mlir_module_serialization_version=self.mlir_module_serialization_version,
        _get_vjp=None,
    )
