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
"""Help list those not-implement ops in jaxonnxruntime for a onnx model.

Example cmd:
List all model zoo model name list:
```
python third_party/py/jaxonnxruntime/tools/analyze_model
```

Find those missing ops in the onnx model.
```
python third_party/py/jaxonnxruntime/tools/analyze_model \
--model_path=t5-decoder-with-lm-head
```
Notes: The model path can be real path name or model zoo onnx model name from
onnx.hub.
"""
from absl import app
from absl import flags
from absl import logging
from jaxonnxruntime.core import call_onnx
from onnx import hub
import onnx
from onnx import helper as onnx_helper

_MODEL_PATH = flags.DEFINE_string(
    'model_path',
    None,
    'The onnx model path or name in onnx model zoo',
)


def get_model_name_list():
  """lists all of the ONNX ops that are implemented in Jaxonnxruntime."""
  return [x.model.lower() for x in hub.list_models()]


def main(argv):
  """Find those not-implement ops for the specific onnx model."""
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  # get the version list for the ONNX operator
  model_path = _MODEL_PATH.value
  if not model_path:
    print(f'model list in zoo: {get_model_name_list()}')
    return
  if model_path in get_model_name_list():
    logging.info(
        'Find the %s model in onnx model zoo, here will use onnx.hub to'
        ' load it',
        model_path,
    )
    onnx_model = hub.load(model_path)
  else:
    onnx_model = onnx.load(model_path)

  if isinstance(onnx_model, onnx.ModelProto):
    op_types = {node.op_type for node in onnx_model.graph.node}
    opset = onnx_helper.make_opsetid(
        onnx.defs.ONNX_DOMAIN, onnx.defs.onnx_opset_version()
    )
    all_impl_op_dict = call_onnx._get_all_handlers([opset])[opset.domain]  # pylint: disable=protected-access
    logging.info(
        'All onnx implmented ops %s is %s', model_path, all_impl_op_dict.keys()
    )
    not_impl_ops = {op for op in op_types if op not in all_impl_op_dict}
    logging.info('Those not-implement ops: %s', not_impl_ops)
  else:
    logging.info('Fail to load model %s', model_path)


if __name__ == '__main__':
  app.run(main)
