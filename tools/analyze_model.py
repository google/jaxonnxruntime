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
import argparse
import onnx
from onnx.helper import make_opsetid
import jaxonnxruntime
from jaxonnxruntime import call_onnx

"""Help list those not-implement ops in jaxonnxruntime for a onnx model."""


def main(args):
  # get the version list for the ONNX operator
  model_path = args.model_path
  onnx_model = onnx.load(model_path)

  op_types = {node.op_type for node in onnx_model.graph.node}
  opset = make_opsetid(onnx.defs.ONNX_DOMAIN, 1)
  all_impl_op_dict = call_onnx._get_all_handlers([opset])[opset.domain]

  not_impl_ops = {op for op in op_types if op not in all_impl_op_dict}
  print(f'Not_Implement ops: {not_impl_ops}')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='list those not-implement ONNX ops.'
  )
  parser.add_argument('model_path', type=str, help='input model path')
  main(parser.parse_args())
