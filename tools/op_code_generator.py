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

"""Help simplify the onnx op develoment, example cmd
`python op_code_generator.py Add`"""
import argparse
import jinja2
import onnx

# define the template for the operator implementation
template = """
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
\"\"\"Define ONNX {{ op_name }} operator.\"\"\"
from collections.abc import Callable
import functools
from typing import Any


from jax import jit
from jax import lax

from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node

@handler.register_op("{{ op_name }}")
class {{ op_name }}(handler.Handler):
  \"\"\"Implementation of the ONNX {{ op_name }} operator.\"\"\"

{% for version in versions %}
  @classmethod
  def version_{{ version }}(cls, node: onnx_node.OnnxNode) -> Callable[..., Any]:
    return onnx_{{ op_name|lower }}
{% endfor %}


@jit
def onnx_{{ op_name|lower}}(x):
  #TODO: add the implementation.
"""
template_obj = jinja2.Template(template)

def main(args):
  # get the version list for the ONNX operator
  op_name = args.op_name
  schema = onnx.defs.get_schema(op_name)
  versions = [schema.since_version] if schema else []

  # render the template with the command line arguments and version list
  values = {
      'op_name': op_name,
      'versions': versions,
  }

  code = template_obj.render(values)
  print(code)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Generate ONNX op Python code.')
  parser.add_argument('op_name', type=str, help='name of the ONNX op')

  # parse command line arguments
  args = parser.parse_args()
  main(args)