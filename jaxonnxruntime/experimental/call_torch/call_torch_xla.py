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

"""The call_torch API  on the pytorch exported stablehlo module."""

from typing import Any, Tuple, Union
from absl import logging
from jax import core
import jax.extend as jex
from jax.interpreters import mlir
from jax.interpreters import xla
from jax.lib import xla_extension
import jax.numpy as jnp
from jaxlib.mlir import ir
from mlir.dialects import func

__all__ = ["call_torch_xla"]


def _clean_mhlo_attributes(mlir_module):
  """Removes those extra mhlo attributes not belong to stablehlo dialect."""

  def walk_stablehlo_operations(op, cb):
    """walk the stablehlo operation recursive with callback function."""
    cb(op)
    for region in op.operation.regions:
      for block in region:
        for op in block:
          walk_stablehlo_operations(op, cb)

  def remove_attribute(op):
    attributes = op.operation.attributes
    # Remove all attributes whole name start with "mhlo." i.e.
    # mhlo.is_dynamicing.
    remove_key_names = [
        key.name for key in attributes if key.name.startswith("mhlo.")
    ]
    for key_name in remove_key_names:
      del op.operation.attributes[key_name]

  new_module = mlir_module
  walk_stablehlo_operations(new_module, remove_attribute)
  return new_module


call_torch_xla_p = jex.core.Primitive("call_torch_xla")
call_torch_xla_p.multiple_results = True


def call_torch_xla(*args, module: Union[str, Any], clean_mhlo_attributes=True):
  """Lower torch module to XLA and wrap it as JAX funtion.

  Given the the torch module and its input arguments, this function will lower
  the torch module into stablehlo first. Then it wrap the stablehlo module as
  JAX function. So it can work with other JAX functions.

  Args:
    *args: The arguments to the called module.
    module: The module to call, it supports one of `torch.nn.Module`, or
      stablehlo module str.
    clean_mhlo_attributes: Remove Mhlo attributes from the stablehlo MlirModule.
      This is typically done when people legalize the compiled Hlo to stablehlo.

  Returns:
    The output of the torch function.
  """
  if not isinstance(module, str):
    raise NotImplementedError(
        "call_torch_xla only support stablehlo str currently."
    )

  if clean_mhlo_attributes:
    with mlir.make_ir_context():
      mlir_module = ir.Module.parse(module)
      new_module = _clean_mhlo_attributes(mlir_module)
      module = str(new_module)
      if logging.vlog_is_on(3):
        logging.vlog(
            3, "Mlir module after clean_mhlo_attributes is:\n\n%s", module
        )

  ret = call_torch_xla_p.bind(*args, module=module)
  return tuple(ret)


def call_torch_xla_impl(*args, module):
  return xla.apply_primitive(call_torch_xla_p, *args, module=module)


call_torch_xla_p.def_impl(call_torch_xla_impl)


# See https://github.com/google/jax/blob/main/jax/_src/interpreters/mlir.py#L115
# for reference
def _ir_type_to_dtype(ir_type: ir.Type) -> jnp.dtype:
  """Converts MLIR type to JAX dtype."""
  ir_to_jax = {
      ir.IntegerType.get_signless(1): jnp.bool_,
      ir.IntegerType.get_signless(8): jnp.int8,
      ir.IntegerType.get_signless(16): jnp.int16,
      ir.IntegerType.get_signless(32): jnp.int32,
      ir.IntegerType.get_signless(64): jnp.int64,
      ir.IntegerType.get_unsigned(8): jnp.uint8,
      ir.IntegerType.get_unsigned(16): jnp.uint16,
      ir.IntegerType.get_unsigned(32): jnp.uint32,
      ir.IntegerType.get_unsigned(64): jnp.uint64,
      ir.F16Type.get(): jnp.float16,
      ir.F32Type.get(): jnp.float32,
      ir.F64Type.get(): jnp.float64,
      ir.BF16Type.get(): jnp.bfloat16,
      ir.ComplexType.get(ir.F32Type.get()): jnp.complex64,
      ir.ComplexType.get(ir.F64Type.get()): jnp.complex128,
      ir.Float8E4M3B11FNUZType.get(): jnp.float8_e4m3b11fnuz,
      ir.Float8E4M3FNType.get(): jnp.float8_e4m3fn,
      ir.Float8E5M2Type.get(): jnp.float8_e5m2,
  }
  return ir_to_jax[ir_type]


_UKNOWN_DIM_PREFIX = "call_torch_unknown_dim"


def call_torch_xla_abstract_eval(
    *in_avals: core.ShapedArray, module: str
) -> Tuple[core.ShapedArray, ...]:
  """Abstract evaluation rule."""
  with mlir.make_ir_context():
    stablehlo_module = ir.Module.parse(module)
    symtab = ir.SymbolTable(stablehlo_module.operation)

    # Check we are not reusing existing dimension vars.
    has_polymorphic = False
    for val in in_avals:
      for dim in val.shape:
        if not isinstance(dim, int):
          has_polymorphic = True
          if any(x.startswith(_UKNOWN_DIM_PREFIX) for x in dim.get_vars()):
            raise ValueError(
                "Polymorphic variable name that start with"
                f" `{_UKNOWN_DIM_PREFIX}` are reserved for use by call_torch"
                f" internal for outputs: `{val.shape}`"
            )

    # Map each `dynamic`` dimension to a unique dimension variable because we
    # do not have the information from the avals of the original JAX function.
    # In practice, the output shapes may actually be much more constrained, but
    # the information is not available here.
    dynamic_count = 0
    output_specs = []
    for res in symtab["main"].type.results:
      if any(dim == res.get_dynamic_size() for dim in res.shape):
        out_shape = ", ".join(
            f"{_UKNOWN_DIM_PREFIX}_{(dynamic_count := dynamic_count + 1)}"
            if dim == res.get_dynamic_size()
            else str(dim)
            for dim in res.shape
        )

        assert has_polymorphic, has_polymorphic
        from jax.experimental.export import shape_poly  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

        out_shape = shape_poly.symbolic_shape(out_shape, like=res.shape)  # pylint: disable=protected-access
      else:
        out_shape = res.shape
      output_specs.append(
          core.ShapedArray(out_shape, _ir_type_to_dtype(res.element_type))
      )
    return tuple(output_specs)


call_torch_xla_p.def_abstract_eval(call_torch_xla_abstract_eval)


# Taken from
# github.com/google/jax/blob/main/jax/experimental/jax2tf/jax_export.py#L859
def refine_polymorphic_shapes(
    module: ir.Module, validate_static_shapes: bool
) -> ir.Module:
  """Refine the polymorphic shapes inside a module.

  Given a module with static input shapes, but using dynamic shapes due to
  shape polymorphism, run shape refinement to resolve all the dynamic shapes.

  Args:
    module: A module with static input shapes but dynamic shapes inside.
    validate_static_shapes: Whether to check all shapes are static after
      refinement.

  Returns:
    The refined module.
  """
  refined_module_str = xla_extension.mlir.refine_polymorphic_shapes(
      mlir.module_to_bytecode(module),
      enable_shape_assertions=validate_static_shapes,
      validate_static_shapes=validate_static_shapes,
  )
  context = mlir.make_ir_context()
  with context:
    return ir.Module.parse(refined_module_str)


def call_torch_xla_lowering(ctx: mlir.LoweringRuleContext, *args, module: str):
  """Lowering rule."""
  program_name = "_call_torch_xla_fn"
  stablehlo_module = ir.Module.parse(module)

  callee_name = mlir.merge_mlir_modules(
      dst_module=ctx.module_context.module,
      sym_name=program_name,
      src_module=stablehlo_module,
  )

  symtab = ir.SymbolTable(ctx.module_context.module.operation)
  result_types = symtab[program_name].type.results

  # Paranoid checks.
  assert len(mlir.flatten_lowering_ir_args(args)) == len(args), (
      len(mlir.flatten_lowering_ir_args(args)),
      len(args),
  )

  call = func.CallOp(
      result_types,
      ir.FlatSymbolRefAttr.get(callee_name),
      args,
  )
  return tuple(x for x in call.results)


mlir.register_lowering(call_torch_xla_p, call_torch_xla_lowering)
