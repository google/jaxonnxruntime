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
"""Help list those not-implement ops in jaxonnxruntime for a onnx model."""
import argparse
import logging
from jaxonnxruntime import call_onnx
import onnx
from onnx import hub
from onnx.helper import make_opsetid


def get_model_name_list():
  """lists all of the ONNX ops that are implemented in Jaxonnxruntime."""
  return [x.model.lower() for x in hub.list_models()]


def main(args):
  """Find those not-implement ops for the specific onnx model."""
  # get the version list for the ONNX operator
  model_path = args.model_path
  if model_path in get_model_name_list():
    onnx_model = hub.load(model_path)
  else:
    onnx_model = onnx.load(model_path)

  if isinstance(onnx_model, onnx.ModelProto):
    op_types = {node.op_type for node in onnx_model.graph.node}
    opset = make_opsetid(onnx.defs.ONNX_DOMAIN, 1)
    all_impl_op_dict = call_onnx._get_all_handlers([opset])[opset.domain]  # pylint: disable=protected-access
    logging.info(
        'All ops used in model %s is %s', model_path, all_impl_op_dict.keys()
    )
    not_impl_ops = {op for op in op_types if op not in all_impl_op_dict}
    logging.info('Those not-implement ops: %s', not_impl_ops)
  else:
    logging.info('Fail to load model %s', model_path)


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  parser = argparse.ArgumentParser(
      description='List those not-implement ONNX ops.'
  )
  parser.add_argument('model_path', type=str, help='input model path')
  main(parser.parse_args())
