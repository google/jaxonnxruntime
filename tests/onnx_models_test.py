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

"""ONNX Model Zoo test."""

import collections
from typing import Any, Optional, Union
import unittest

from absl import app
from absl import logging
from absl.testing import absltest
import jax
import jaxonnxruntime as jort
from jaxonnxruntime import runner
from jaxonnxruntime.backend import Backend as JaxBackend  # pylint: disable=g-importing-member
from jaxonnxruntime.core import onnx_utils
from onnx import hub
import numpy as np

import onnx


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
          model_info.metadata.get('io_ports', {}).get('inputs')
      )
      model_inputs = [_create_dummy_tensor(item) for item in model_inputs_info]
      test_case = onnx_utils.JortTestCase()
      test_case.assert_model_run_through(model, model_inputs)  # pytype: disable=wrong-arg-types
      test_case.assert_ort_jort_all_close(model, model_inputs)  # pytype: disable=wrong-arg-types

    model_test_name = model_name.replace('-', '_').replace(' ', '_')
    model_test_name = f'test_{model_test_name}'
    self._add_test(kind, model_test_name, run, model_marker)

  def check_compatibility(self, model: onnx.ModelProto, device: str):
    if (
        hasattr(self.backend, 'is_compatible')
        and callable(self.backend.is_compatible)
        and not self.backend.is_compatible(model, device)
    ):
      raise unittest.SkipTest(f'Not compatible with backend with {device}')


def main(unused_argv):
  backend_test = Runner(JaxBackend, __name__)
  expect_fail_patterns = []
  include_patterns = []
  exclude_patterns = []

  exclude_patterns.extend([
      "test_caffenet_int8_cpu",  # There some ORT ops not covered by ONNX.
      "test_t5_decoder_with_lm_head_cpu",
      "test_yolov3_12_int8_cpu",
      "test_fcn_resnet_50_int8_cpu",
      "test_yolov3_12_cpu",
      "test_alexnet_int8_cpu",
      "test_vgg_16_int8_cpu",
      "test_candy_cpu",
      "test_googlenet_int8_cpu",
      "test_shufflenet_v2_int8_cpu",
      "test_yolov4_cpu",
      "test_resnet50_qdq_cpu",
      "test_faster_r_cnn_r_50_fpn_cpu",
      "test_ssd_mobilenetv1_12_cpu",
      "test_mnist_12_int8_cpu",
      "test_mobilenet_v2_1",
      "test_efficientnet_lite4_cpu",
      "test_squeezenet_1",
      "test_faster_r_cnn_r_50_fpn_qdq_cpu",
      "test_mask_r_cnn_r_50_fpn_fp32_cpu",
      "test_resnet101_duc_hdc_12_int8_cpu",
      "test_alexnet_qdq_cpu",
      "test_faster_r_cnn_r_50_fpn_fp32_cpu",
      "test_lresnet100e_ir_cpu",
      "test_vgg_16_qdq_cpu",
      "test_caffenet_qdq_cpu",
      "test_t5_encoder_cpu",
      "test_resnet_preproc_cpu",
      "test_inception_1_qdq_cpu",
      "test_efficientnet_lite4_qdq_cpu",
      "test_mosaic_cpu",
      "test_lresnet100e_ir_int8_cpu",
      "test_squeezenet_1",
      "test_bidaf_int8_cpu",
      "test_pointilism_cpu",
      "test_resnet50_int8_cpu",
      "test_ssd_int8_cpu",
      "test_retinanet_",
      "test_ssd_mobilenetv1_cpu",
      "test_fcn_resnet_50_qdq_cpu",
      "test_fcn_resnet_101_cpu",
      "test_mask_r_cnn_r_50_fpn_int8_cpu",
      "test_roberta_base_cpu",
      "test_ssd_qdq_cpu",
      "test_inception_1_int8_cpu",
      "test_mobilenet_v2_1",
      "test_googlenet_qdq_cpu",
      "test_mobilenet_v2_1",
      "test_shufflenet_v2_qdq_cpu",
      "test_densenet_121_12_int8_cpu",
      "test_udnie_cpu",
      "test_zfnet_512_int8_cpu",
      "test_bert_squad_int8_cpu",
      "test_mask_r_cnn_r_50_fpn_cpu",
      "test_ssd_cpu",
      "test_fcn_resnet_50_cpu",
      "test_mask_r_cnn_r_50_fpn_qdq_cpu",
      "test_roberta_sequenceclassification_cpu",
      "test_rain_princess_cpu",
      "test_ssd_mobilenetv1_12_int8_cpu",
      "test_bidaf_cpu",
      "test_tiny_yolov3_cpu",
      "test_yolov3_cpu",
      "test_faster_r_cnn_r_50_fpn_int8_cpu",
      "test_ssd_mobilenetv1_12_qdq_cpu",
      "test_zfnet_512_qdq_cpu",
      "test_efficientnet_lite4_int8_cpu",
  ])

  for pattern in include_patterns:
    backend_test.include(pattern)

  for pattern in exclude_patterns:
    backend_test.exclude(pattern)

  for pattern in expect_fail_patterns:
    backend_test.xfail(pattern)

  # import all test cases at global scope to make them visible to
  # python.unittest
  logging.info(
      'list all test_cases: %s', '\n'.join(backend_test.test_cases.keys())
  )
  globals().update(backend_test.test_cases)
  absltest.main()


if __name__ == '__main__':
  app.run(main)
