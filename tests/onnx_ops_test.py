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

"""ONNX node test."""

import collections
from typing import Any
from absl.testing import absltest
from jaxonnxruntime import runner
from jaxonnxruntime.backend import Backend as JaxBackend


class Runner(runner.Runner):

  def __init__(self, backend: JaxBackend, parent_module: Any = None) -> None:
    self.backend = backend
    self._parent_module = parent_module
    self._include_patterns = set()  # type: ignore[var-annotated]
    self._exclude_patterns = set()  # type: ignore[var-annotated]
    self._xfail_patterns = set()  # type: ignore[var-annotated]
    self._test_items = collections.defaultdict(dict)  # type: ignore[var-annotated]

    for rt in runner.load_model_tests(kind='node'):
      self._add_model_test(rt, 'Node')


class NodeTest(absltest.TestCase):
  pass


backend_test = Runner(JaxBackend, __name__)
expect_fail_patterns = []
include_patterns = []
exclude_patterns = []

include_patterns.append('test_abs_')
include_patterns.append('test_add_')
include_patterns.append('test_cast_')
include_patterns.append('test_concat_')
include_patterns.append('test_constantofshape_')
include_patterns.append('test_conv_')
include_patterns.append('test_div_')
include_patterns.append('test_gather_')
include_patterns.append('test_gemm_')
include_patterns.append('test_matmul_')
include_patterns.append('test_mul_')
include_patterns.append('test_nonzero_')

# TODO(johnqiangzhang): should modify onnx.numpy_helper.to_array to support load
# bfloat16.
exclude_patterns.append('test_cast_FLOAT_to_BFLOAT16')
# Not implement yet
exclude_patterns.append('test_gather_elements_')


expect_fail_patterns.append('test_cast_FLOAT_to_STRING')
expect_fail_patterns.append('test_cast_STRING_to_FLOAT')


for pattern in include_patterns:
  backend_test.include(pattern)

for pattern in exclude_patterns:
  backend_test.exclude(pattern)

for pattern in expect_fail_patterns:
  backend_test.xfail(pattern)

for name, func in backend_test.test_cases.items():
  setattr(NodeTest, name, func)


if __name__ == '__main__':
  absltest.main()
