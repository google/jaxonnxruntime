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
from jaxonnxruntime import onnx_ops
import onnx


# define the template for the operator implementation
template_head = """# Copyright 2023 The Jaxonnxruntime Authors.
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
\"\"\"Define ONNX {op_name} operator.\"\"\"
# pylint: disable=unused-argument
# pylint: disable=g-explicit-length-test
import functools
import inspect
from collections.abc import Callable, Sequence
from typing import Any

from jax import jit
from jax import numpy as jnp
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node


@handler.register_op("{op_name}")
class {op_name}(handler.Handler):
  \"\"\"Implementation of the ONNX {op_name} operator.\"\"\"

  @classmethod
  def _prepare(cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any):
    sig = inspect.signature(onnx_jax_impl)
    kwparams = [param.name for param in sig.parameters.values() if param.kind == inspect.Parameter.KEYWORD_ONLY]
    for name in kwparams:
      node.attrs_dict[name] = node.attrs.get(name, None)
"""

template_version_func = """
  @classmethod
  def version_{version}(cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]) -> Callable[..., Any]:
    \"\"\"ONNX version_{version} {op_name} op.\"\"\"
    cls._prepare(node, inputs, onnx_{op_name_lower})
    return onnx_{op_name_lower}
"""

template_tail = """

@functools.partial(jit, static_argnames=())
def onnx_{op_name_lower}(*input_args):
  \"\"\"https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#{op_name} for more details.\"\"\"
  # TODO({username}): add the implementation here.
  # Then update the onnx_ops_teset.py to include it,
  # `include_patterns.append('test_{op_name_lower}_')`.
  return input_args
"""

root_dir = os.path.dirname(os.path.realpath(onnx_ops.__file__))
op_schema_set = {
    str(op_schema.name) for op_schema in onnx.defs.get_all_schemas()
}


def update_onnx_ops_init_file(op_name):
  """Update onnx_ops/__init_.py with the created op."""
  init_py_file = os.path.join(root_dir, '__init__.py')
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
  if str(op_name) not in op_schema_set:
    raise ValueError(
        f'ONNX {op_name} is not ONNX op list {sorted(op_schema_set)}?.'
    )
  schema = onnx.defs.get_schema(op_name)
  versions = [schema.since_version] if schema else []
  username = os.environ['USER']

  # Render the template and create new op file under onnx_ops folder.
  code = template_head.format(
      op_name=op_name, op_name_lower=op_name.lower(), username=username
  )
  for version in versions:
    code += template_version_func.format(
        version=version, op_name_lower=op_name.lower(), op_name=op_name
    )
  code += template_tail.format(
      op_name_lower=op_name.lower(),
      op_name=op_name,
      username=username,
  )
  logging.info('Genereate new code=\n%s', code)
  op_def_path = os.path.join(root_dir, f'{op_name.lower()}.py')
  with open(op_def_path, 'w') as f:
    f.write(code)

  # Update the onnx_ops/__init__.py by adding this new op.
  update_onnx_ops_init_file(op_name)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Generate ONNX op Python code.')
  parser.add_argument('op_name', type=str, help='name of the ONNX op')
  main(parser.parse_args())
