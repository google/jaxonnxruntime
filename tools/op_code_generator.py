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

"""Help simplify the onnx op develoment, example cmd."""
# Example cmd: `python op_code_generator.py Add`
import argparse
import logging
import os
import re
from jinja2 import Template
import onnx

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
root_dir = os.path.join(root_dir, 'jaxonnxruntime')
op_schema_dict = {
    str(op_schema.name): op_schema for op_schema in onnx.defs.get_all_schemas()
}

# define the template for the operator implementation
template = Template("""\
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
\"\"\"Define ONNX {{op_name}} operator.\"\"\"
# pylint: disable=unused-argument
# pylint: disable=g-explicit-length-test
from collections.abc import Callable, Sequence
import functools
import inspect
from typing import Any

import jax
from jax import jit
from jax import numpy as jnp
from jax._src.interpreters import mlir
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node
from jaxonnxruntime.core import onnx_primitive


@handler.register_op("{{op_name}}")
class {{op_name}}(handler.Handler):
  \"\"\"Implementation of the ONNX {{op_name}} operator.\"\"\"

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    sig = inspect.signature(onnx_jax_impl)
    kwparams = [
        param.name
        for param in sig.parameters.values()
        if param.kind == inspect.Parameter.KEYWORD_ONLY
    ]
    for name in kwparams:
      node.attrs_dict[name] = node.attrs.get(name, None)
{% for version in versions %}
  @classmethod
  def version_{{ version }}(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    \"\"\"ONNX version {{version}} {{ op_name }} op.\"\"\"
    cls._prepare(node, inputs, onnx_{{ op_name|lower }})
    return onnx_{{ op_name|lower }}
{% endfor %}

@functools.partial(jit, static_argnames=({{static_arg_attr_list}}))
def onnx_{{op_name|lower}}(*args{{attr_list}}):
  \"\"\"https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#{{op_name}} for more details.\"\"\"
{% if min_input != max_input %}
  if len(args) < {{min_input}} or len(args) > {{max_input}}:
    raise ValueError(
        f"len(args) should be within [{{min_input}}, {{max_input}}] but got {len(args)}"
    )
    all_args = args + [None] * ({{inputs_name|length}} - len(args))
{% else %}
  if len(args) != {{min_input}}:
    raise ValueError(
        f"len(args) should equal to {{min_input}} but got {len(args)}"
    )
  all_args = args
{% endif %}
  return onnx_{{op_name|lower}}_p.bind(*all_args)

# Define onnx_{{op_name|lower}}_p primitive.
onnx_{{op_name|lower}}_p = onnx_primitive.OnnxPrimitive("onnx_{{op_name|lower}}")
onnx_{{op_name|lower}}_p.multiple_results = False


@onnx_{{op_name|lower}}_p.def_impl
def _onnx_{{op_name|lower}}_impl(*args):
  # TODO({{username}}): add the implementation here.
  # Then update the onnx_ops_teset.py to include it,
  # `include_patterns.append('test_{{op_name|lower}}_')`.
  return


@onnx_{{op_name|lower}}_p.def_abstract_eval
def _onnx_{{op_name|lower}}_abstract_eval(*args):
  aval_args = jax.tree_map(
      lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), args
  )
  out = jax.eval_shape(_onnx_abs_impl, *aval_args)
  return jax.tree_map(
      lambda x: jax.abstract_arrays.ShapedArray(x.shape, x.dtype), out
  )


def _onnx_{{op_name|lower}}_lowering(ctx, *args, platform):
  \"\"\"{{op_name|lower}} lowering rule.\"\"\"
  jit_func = jax.jit(_onnx_{{op_name|lower}}_impl)
  jit_func_lowering = mlir.lower_fun(jit_func, onnx_{{op_name|lower}}_p.multiple_results)
  return mlir.delegate_lowering(ctx, jit_func_lowering, *args)


for _p in ("cpu", "tpu", "cuda", "rocm"):
  mlir.register_lowering(
      onnx_{{op_name|lower}}_p,
      functools.partial(_onnx_{{op_name|lower}}_lowering, platform=_p),
      platform=_p,
  )

""")


def create_op_schema_render_dict(op_name):
  """Create the render_dict for the template."""
  username = os.environ['USER']
  assert op_name in op_schema_dict, f'{op_name} is not legal ONNX op name.'
  op_schema = op_schema_dict[op_name]
  versions = [op_schema.since_version] if op_schema else []
  render_dict = {
      'op_name': op_name,
      'username': username,
      'min_input': op_schema.min_input,
      'max_input': op_schema.max_input,
      'min_output': op_schema.min_output,
      'max_output': op_schema.max_output,
      'attribute_list': list(op_schema.attributes.keys()),
      'deprecated': op_schema.deprecated,
      'doc': op_schema.doc,
      'domain': op_schema.domain,
      'versions': versions,
      'inputs_name': [i.name.lower() for i in op_schema.inputs],
  }
  attr_list = list(op_schema.attributes.keys())
  render_dict['attr_list'] = (
      (', ' + ', '.join(attr_list)) if len(attr_list) else ''
  )
  render_dict['static_arg_attr_list'] = ', '.join(
      ["'" + name + "'" for name in attr_list]
  )
  return render_dict


def update_onnx_ops_init_file(op_name):
  """Update onnx_ops/__init_.py with the created op."""
  init_py_file = os.path.join(root_dir, 'onnx_ops/__init__.py')
  with open(init_py_file, 'r') as f:
    existing_imports = f.read()

  new_import = f'from . import {op_name.lower()}'
  if new_import in existing_imports:
    logging.info('Already have %s in onnx_ops/__init__.py.', op_name)
    return

  # Split the existing imports.
  initial_imports = existing_imports.split('from . import ')[0]
  onnx_op_imports = existing_imports[len(initial_imports) :]

  # Add the new import statement.
  onnx_op_imports += f'\nfrom . import {op_name.lower()}'

  # Replace and sort the ONNX op imports.
  pattern = r'from \. import .*'
  sorted_imports = sorted(re.findall(pattern, onnx_op_imports))
  sorted_imports_str = '\n'.join(sorted_imports)

  # Write the sorted import statements to the file
  with open(init_py_file, 'w') as f:
    f.write(initial_imports + sorted_imports_str)
    f.write('\n')


def main(args):
  # get the version list for the ONNX operator
  op_name = args.op_name
  if str(op_name) not in op_schema_dict:
    raise ValueError(
        f'ONNX {op_name} is not valid ONNX op',
        f'see full list {sorted(op_schema_dict.keys())}.',
    )
  render_dict = create_op_schema_render_dict(op_name)
  code = template.render(**render_dict)
  logging.info('Genereate new code=\n%s', code)
  op_def_path = os.path.join(root_dir, f'onnx_ops/{op_name.lower()}.py')
  with open(op_def_path, 'w') as f:
    f.write(code)

  # Update the onnx_ops/__init__.py by adding this new op.
  update_onnx_ops_init_file(op_name)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Generate ONNX op Python code.')
  parser.add_argument('op_name', type=str, help='name of the ONNX op')
  main(parser.parse_args())
