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

"""ONNX Model Zoo test."""

import collections
from typing import Any, Optional, Union
import unittest

from absl import flags
from absl import logging
from absl.testing import absltest
import jax
import jaxonnxruntime as jort
from jaxonnxruntime import runner
from jaxonnxruntime.backend import Backend as JaxBackend  # pylint: disable=g-importing-member
from jaxonnxruntime.core import onnx_utils
import numpy as np

import onnx
from onnx import hub

jax.config.update('jax_enable_x64', True)
jax.config.update('jax_numpy_rank_promotion', 'warn')
jort.config.update('jaxort_only_allow_initializers_as_static_args', False)
jort.config.update('jaxort_nonzero_use_fully_padding', True)


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


class Runner(runner.Runner):

  def __init__(
      self, backend: type(JaxBackend), parent_module: Any = None
  ) -> None:
    self.backend = backend
    self._parent_module = parent_module
    self._include_patterns = set()  # type: ignore[var-annotated]
    self._exclude_patterns = set()  # type: ignore[var-annotated]
    self._xfail_patterns = set()  # type: ignore[var-annotated]
    self._test_items = collections.defaultdict(dict)  # type: ignore[var-annotated]

    model_name_list = set(map(lambda x: x.model.lower(), hub.list_models()))
    for name in model_name_list:
      self._add_model_test(name, 'ModelZoo')

  def _add_model_test(self, model_name: str, kind: str) -> None:
    """Add model test cases."""
    model_marker: list[Optional[Union[onnx.ModelProto, onnx.NodeProto]]] = [
        None
    ]

    def run(test_self: Any, device: str) -> None:  # pylint: disable=unused-argument
      if logging.vlog_is_on(3):
        logging.vlog(3, f'jax devices = {jax.devices()}')
        logging.vlog(3, f'default backend = {jax.default_backend()}')
      model = hub.load(model_name)
      model_marker[0] = model
      self.check_compatibility(model, device)
      model_info = hub.get_model_info(model_name)
      model_inputs_info = list(
          model_info.metadata.get('io_ports').get('inputs')
      )
      model_inputs = [_create_dummy_tensor(item) for item in model_inputs_info]
      test_case = onnx_utils.JortTestCase()
      test_case.assert_model_run_through(model, model_inputs)
      test_case.assert_ort_jort_all_close(model, model_inputs)

    model_test_name = model_name.replace('-', '_').replace(' ', '_')
    model_test_name = f'test_{model_test_name}'
    self._add_test(kind, model_test_name, run, model_marker)

  def check_compatibility(self, model: onnx.ModelProto, device: str):
    if (
        hasattr(self.backend, 'is_compatible')
        and callable(self.backend.is_compatible)
        and not self.backend.is_compatible(model, device)
    ):
      raise unittest.SkipTest('Not compatible with backend')


backend_test = Runner(JaxBackend, __name__)
expect_fail_patterns = []
include_patterns = []
exclude_patterns = []

include_patterns.append('test_alexnet_')
include_patterns.append('test_bert_squad_')
include_patterns.append('test_caffenet_')
include_patterns.append('test_densenet_121_12_')
include_patterns.append('test_emotion_ferplus_')
include_patterns.append('test_googlenet_')
include_patterns.append('test_gpt_2_')
include_patterns.append('test_gpt_2_lm_head_')
include_patterns.append('test_inception_1_')
include_patterns.append('test_inception_2_')
include_patterns.append('test_mnist_12_')
include_patterns.append('test_mnist_')
include_patterns.append('test_r_cnn_ilsvrc13_')
include_patterns.append('test_resnet101_')
include_patterns.append('test_resnet101_duc_hdc_12_')
include_patterns.append('test_resnet101_duc_hdc_')
include_patterns.append('test_resnet101_v2_')
include_patterns.append('test_resnet152_')
include_patterns.append('test_resnet152_v2_')
include_patterns.append('test_resnet18_')
include_patterns.append('test_resnet18_v2_')
include_patterns.append('test_resnet34_')
include_patterns.append('test_resnet34_v2_')
include_patterns.append('test_resnet50_caffe2_')
include_patterns.append('test_resnet50_')
include_patterns.append('test_resnet50_fp32_')
include_patterns.append('test_resnet50_v2_')
include_patterns.append('test_shufflenet_v1_')
include_patterns.append('test_shufflenet_v2_')
include_patterns.append('test_shufflenet_v2_fp32_')
include_patterns.append('test_squeezenet_1.0_')
include_patterns.append('test_squeezenet_1.1_')
include_patterns.append('test_super_resolution_')
include_patterns.append('test_tiny_yolov2_')
include_patterns.append('test_vgg_16_bn_')
include_patterns.append('test_vgg_16_')
include_patterns.append('test_vgg_16_fp32_')
include_patterns.append('test_vgg_19_bn_')
include_patterns.append('test_vgg_19_caffe2_')
include_patterns.append('test_vgg_19_')
include_patterns.append('test_yolov2_')
include_patterns.append('test_zfnet_512._')

for pattern in include_patterns:
  backend_test.include(pattern)

for pattern in exclude_patterns:
  backend_test.exclude(pattern)

for pattern in expect_fail_patterns:
  backend_test.xfail(pattern)

# import all test cases at global scope to make them visible to python.unittest
globals().update(backend_test.test_cases)


if __name__ == '__main__':
  absltest.main()
