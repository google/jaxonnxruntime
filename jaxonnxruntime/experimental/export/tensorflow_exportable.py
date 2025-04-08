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

"""Tensorflow Exportable class."""

import dataclasses

import jax
from jax.lib import xla_extension
from jaxonnxruntime.experimental.export import exportable
from jaxonnxruntime.experimental.export import exportable_utils
import tensorflow as tf

Tensor = tf.Tensor
Sharding = exportable_utils.Sharding
HloSharding = exportable_utils.HloSharding


@dataclasses.dataclass
class TensorWithSharding:
  tensor: Tensor
  sharding: Sharding = None


@dataclasses.dataclass
class TensorflowExportable(exportable.Exportable):
  """Tensorflow Exportable class."""

  _out_hlo_sharding: tuple[HloSharding, ...] | None = None

  @classmethod
  def _to_xla_hlo_sharding(
      cls,
      sharding: jax.sharding.Sharding,
      tensor: Tensor,
  ) -> HloSharding:
    if sharding is None:
      return None
    ndim = tf.experimental.numpy.ndim(tensor)
    return sharding._to_xla_hlo_sharding(ndim)  # pylint: disable=protected-access

  def __post_init__(self):
    self.args_maybe_sharding_flat, self._in_tree = jax.tree_util.tree_flatten(
        (self.args, self.kwargs)
    )

    def callable_flat_tf(*args_flat):
      args, kwargs = jax.tree_util.tree_unflatten(self.in_tree, args_flat)
      return self.function(*args, **kwargs)

    self.function_flat_tf = tf.function(callable_flat_tf, jit_compile=True)
    args_flat = jax.tree_util.tree_map(
        lambda x: x.tensor if isinstance(x, TensorWithSharding) else x,
        self.args_maybe_sharding_flat,
    )
    self.concrete_function_tf = self.function_flat_tf.get_concrete_function(
        *args_flat
    )
    _, self._out_tree = jax.tree_util.tree_flatten(
        self.concrete_function_tf.structured_outputs
    )

  @property
  def in_tree(self):
    return self._in_tree

  @property
  def out_tree(self):
    return self._out_tree

  @property
  def tf_platform(self) -> str:
    assert len(self.platforms) == 1
    lowering_platform = self.platforms[0]

    if lowering_platform in ["cpu", "tpu"]:
      return lowering_platform.upper()
    elif lowering_platform == "cuda":
      return "GPU"
    else:
      raise ValueError("platform {platform} not supported")

  @property
  def fun_name(self) -> str:
    return getattr(self.function, "__name__", "unknown")

  @property
  def in_avals(self) -> tuple[jax.core.ShapedArray, ...]:
    def _to_jax_abstract_shaped_array(x):
      if isinstance(x, TensorWithSharding):
        x = x.tensor
      return exportable_utils.tf_tensor_to_jax_abstract_shaped_array(x)

    args_avals_flat = jax.tree_util.tree_map(
        _to_jax_abstract_shaped_array,
        self.args_maybe_sharding_flat,
    )
    return tuple(args_avals_flat)

  @property
  def out_avals(self) -> tuple[jax.core.ShapedArray, ...]:
    out_avals_flat = jax.tree_util.tree_map(
        exportable_utils.tf_tensor_to_jax_abstract_shaped_array,
        self.concrete_function_tf.outputs,
    )
    return tuple(out_avals_flat)

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
  def mlir_module_str(self) -> bytes:
    """Returns the mlir module from TF."""
    args_tf_flat = jax.tree_util.tree_map(
        lambda x: x.tensor if isinstance(x, TensorWithSharding) else x,
        self.args_maybe_sharding_flat,
    )
    func_tf_hlo = self.function_flat_tf.experimental_get_compiler_ir(
        *args_tf_flat
    )(stage="hlo_serialized", platform_name=self.tf_platform)

    mlir_str = xla_extension.mlir.hlo_to_stablehlo(func_tf_hlo)
    return mlir_str

  @property
  def mlir_module_serialized(self) -> bytes:
    return exportable_utils.serialize_stablehlo_mlir_str(self.mlir_module_str)

  @property
  def ordered_effects(self):
    return tuple()

  @property
  def unordered_effects(self):
    return tuple()
