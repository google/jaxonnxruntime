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

import json

from absl.testing import absltest
from jaxonnxruntime import backend as jort_backend
from jaxonnxruntime import config
import numpy as np

import onnx


from onnx import hub


def _load_model(model_name, *_):
  model = hub.load(model_name)
  model = onnx.shape_inference.infer_shapes(model)
  model_info = hub.get_model_info(model_name)
  model_inputs_info = list(model_info.metadata.get('io_ports').get('inputs'))
  return model, model_inputs_info


def _get_tensor_type_name(s_type):
  split_str = s_type.split('(')
  split_s = split_str[1].split(')')
  if len(split_s) > 2:
    raise NotImplementedError('Encountered multiple Tensor types!')
  return split_s[0]


def _create_dummy_tensor(model_info_input):
  shape = model_info_input.get('shape')
  for i, _ in enumerate(shape):
    if isinstance(shape[i], str):
      shape[i] = 1
  dtype_name = _get_tensor_type_name(model_info_input.get('type'))
  dtype_value = onnx.TensorProto.DataType.Value(dtype_name.upper())
  dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[dtype_value]
  return np.random.normal(size=shape).astype(dtype)


def _run_model_test(model_name, model_dir):
  model, model_inputs_info = _load_model(
      model_name,
      model_dir,
  )
  inputs = [_create_dummy_tensor(item) for item in model_inputs_info]
  _ = jort_backend.run(model, inputs)


class TestModelRunThrough(absltest.TestCase):

  def test_bertsquad_12(self):
    model_name = 'bert-squad'
    model_dir = None
    prev_jaxort_only_allow_initializers_as_static_args = (
        config.jaxort_only_allow_initializers_as_static_args
    )
    config.update('jaxort_only_allow_initializers_as_static_args', False)
    _run_model_test(model_name, model_dir)
    config.update(
        'jaxort_only_allow_initializers_as_static_args',
        prev_jaxort_only_allow_initializers_as_static_args,
    )

  def test_gpt2_10(self):
    model_name = 'gpt-2'
    model_dir = None
    prev_jaxort_nonzero_use_fully_padding = (
        config.jaxort_nonzero_use_fully_padding
    )
    config.update('jaxort_nonzero_use_fully_padding', True)
    prev_jaxort_only_allow_initializers_as_static_args = (
        config.jaxort_only_allow_initializers_as_static_args
    )
    config.update('jaxort_only_allow_initializers_as_static_args', False)
    _run_model_test(model_name, model_dir)
    config.update(
        'jaxort_nonzero_use_fully_padding',
        prev_jaxort_nonzero_use_fully_padding,
    )
    config.update(
        'jaxort_only_allow_initializers_as_static_args',
        prev_jaxort_only_allow_initializers_as_static_args,
    )

  def test_resnet50_v1_7(self):
    model_name = 'resnet50'
    model_dir = None
    _run_model_test(model_name, model_dir)


if __name__ == '__main__':
  absltest.main()
