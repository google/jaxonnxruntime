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

"""Exported base class."""

from collections.abc import Callable, Sequence
import dataclasses
from typing import Any

import jax
from jax import export as jax_export
from jaxonnxruntime.experimental.export import exportable_utils


HloSharding = exportable_utils.HloSharding


@dataclasses.dataclass
class Exportable:
  """Exportable base class."""

  function: Callable[..., Any]
  args: tuple[Any, ...]
  kwargs: dict[str, Any]
  lowering_platforms: Sequence[str] | None = None
  disabled_checks: Sequence[jax_export.DisabledSafetyCheck] = ()

  @classmethod
  def _to_xla_hlo_sharding(
      cls, s: Any, aval: jax.core.ShapedArray
  ) -> HloSharding:
    if str(s) == "UnspecifiedValue":
      return None
    return s._to_xla_hlo_sharding(aval.ndim)  # pylint: disable=protected-access

  def __post_init__(self):
    if not hasattr(self.function, "trace"):
      wrapped_fun_jax = jax.jit(self.function)
    else:
      wrapped_fun_jax = self.function

    traced = wrapped_fun_jax.trace(*self.args, **self.kwargs)
    self.lowered = traced.lower(
        lowering_platforms=self.platforms,
    )
    self.lowering = self.lowered._lowering  # pylint: disable=protected-access

  @property
  def platforms(self) -> tuple[str, ...]:
    if self.lowering_platforms is not None:
      assert isinstance(self.lowering_platforms, Sequence)
      return tuple(
          self.lowering_platforms,
      )
    else:
      return (jax_export.default_export_platform(),)

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
  def in_shardings_hlo(self) -> tuple[HloSharding, ...]:
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
  def out_shardings_hlo(self) -> tuple[HloSharding, ...]:
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
  def calling_convention_version(self) -> int:
    """Returns the jax_export_calling_convention_version of the mlir module."""
    version = jax.config.jax_export_calling_convention_version
    minimum_supported_calling_convention_version = (
        jax_export.minimum_supported_calling_convention_version
    )
    maximum_supported_calling_convention_version = (
        jax_export.maximum_supported_calling_convention_version
    )
    if (
        version < minimum_supported_calling_convention_version
        or version > maximum_supported_calling_convention_version
    ):
      raise ValueError(
          f"The requested jax_serialization version {version} is outside the "
          "range of supported versions"
          f" [{minimum_supported_calling_convention_version}"
          f"..{maximum_supported_calling_convention_version}]"
      )
    return version

  @property
  def in_tree(self):
    return self.lowered.in_tree

  @property
  def out_tree(self):
    return self.lowered.out_tree

  @property
  def ordered_effects(self):
    if "ordered_effects" in self.lowering.compile_args:
      return tuple(self.lowering.compile_args["ordered_effects"])
    else:
      return tuple()

  @property
  def unordered_effects(self):
    if "unordered_effects" in self.lowering.compile_args:
      return tuple(self.lowering.compile_args["unordered_effects"])
    else:
      return tuple()

  @property
  def disabled_safety_checks(self):
    return tuple(self.disabled_checks)

  @property
  def uses_global_constants(self):
    return True

  def export(self) -> jax_export.Exported:
    return jax_export.Exported(
        fun_name=self.fun_name,
        in_tree=self.in_tree,
        out_tree=self.out_tree,
        in_avals=self.in_avals,
        out_avals=self.out_avals,
        in_shardings_hlo=self.in_shardings_hlo,
        out_shardings_hlo=self.out_shardings_hlo,
        nr_devices=self.nr_devices,
        platforms=self.platforms,
        ordered_effects=self.ordered_effects,
        unordered_effects=self.unordered_effects,
        disabled_safety_checks=self.disabled_safety_checks,
        mlir_module_serialized=self.mlir_module_serialized,
        module_kept_var_idx=self.module_kept_var_idx,
        uses_global_constants=self.uses_global_constants,
        calling_convention_version=self.calling_convention_version,
        _get_vjp=None,
    )
