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
import jax
from jaxonnxruntime import config
from jaxonnxruntime import runner
from jaxonnxruntime.backend import Backend as JaxBackend  # pylint: disable=g-importing-member


jax.config.update('jax_enable_x64', True)
jax.config.update('jax_numpy_rank_promotion', 'allow')
config.update('jaxort_only_allow_initializers_as_static_args', False)


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

    for rt in runner.load_model_tests(kind='node'):
      self._add_model_test(rt, 'Node')

    for rt in runner.load_model_tests(kind='simple'):
      self._add_model_test(rt, 'Simple')

    for ct in runner.load_model_tests(kind='pytorch-converted'):
      self._add_model_test(ct, 'PyTorchConverted')

    for ot in runner.load_model_tests(kind='pytorch-operator'):
      self._add_model_test(ot, 'PyTorchOperator')


class NodeTest(absltest.TestCase):
  pass


backend_test = Runner(JaxBackend, __name__)
expect_fail_patterns = []
include_patterns = []
exclude_patterns = []


expect_fail_patterns.extend([
    # cast
    'test_cast_FLOAT_to_STRING',
    'test_cast_STRING_to_FLOAT',
    'test_cast_FLOAT16_to_FLOAT8E4M3FNUZ_',
    'test_cast_FLOAT16_to_FLOAT8E4M3FNUZ_cpu',
    'test_cast_FLOAT16_to_FLOAT8E4M3FN_cpu',
    'test_cast_FLOAT16_to_FLOAT8E5M2FNUZ_cpu',
    'test_cast_FLOAT16_to_FLOAT8E5M2_cpu',
    'test_cast_FLOAT_to_FLOAT8E4M3FNUZ_cpu',
    'test_cast_FLOAT_to_FLOAT8E4M3FN_cpu',
    'test_cast_FLOAT_to_FLOAT8E5M2FNUZ_cpu',
    'test_cast_FLOAT_to_FLOAT8E5M2_cpu',
    'test_cast_no_saturate_FLOAT16_to_FLOAT8E4M3FNUZ_cpu',
    'test_cast_no_saturate_FLOAT16_to_FLOAT8E4M3FN_cpu',
    'test_cast_no_saturate_FLOAT16_to_FLOAT8E5M2FNUZ_cpu',
    'test_cast_no_saturate_FLOAT16_to_FLOAT8E5M2_cpu',
    'test_cast_no_saturate_FLOAT_to_FLOAT8E4M3FNUZ_cpu',
    'test_cast_no_saturate_FLOAT_to_FLOAT8E4M3FN_cpu',
    'test_cast_no_saturate_FLOAT_to_FLOAT8E5M2FNUZ_cpu',
    'test_cast_no_saturate_FLOAT_to_FLOAT8E5M2_cpu',
    # castlike
    'test_castlike_FLOAT_to_STRING',
    'test_castlike_STRING_to_FLOAT',
    'test_castlike_FLOAT16_to_FLOAT8E4M3FNUZ_',
    'test_castlike_FLOAT16_to_FLOAT8E4M3FNUZ_',
    'test_castlike_FLOAT16_to_FLOAT8E4M3FN_',
    'test_castlike_FLOAT16_to_FLOAT8E5M2FNUZ_',
    'test_castlike_FLOAT16_to_FLOAT8E5M2_',
    'test_castlike_FLOAT_to_FLOAT8E4M3FNUZ_',
    'test_castlike_FLOAT_to_FLOAT8E4M3FN_',
    'test_castlike_FLOAT_to_FLOAT8E5M2FNUZ_',
    'test_castlike_FLOAT_to_FLOAT8E5M2_',
    'test_castlike_no_saturate_FLOAT16_to_FLOAT8E4M3FNUZ_',
    'test_castlike_no_saturate_FLOAT16_to_FLOAT8E4M3FN_',
    'test_castlike_no_saturate_FLOAT16_to_FLOAT8E5M2FNUZ_',
    'test_castlike_no_saturate_FLOAT16_to_FLOAT8E5M2_',
    'test_castlike_no_saturate_FLOAT_to_FLOAT8E4M3FNUZ_',
    'test_castlike_no_saturate_FLOAT_to_FLOAT8E4M3FN_',
    'test_castlike_no_saturate_FLOAT_to_FLOAT8E5M2FNUZ_',
    'test_castlike_no_saturate_FLOAT_to_FLOAT8E5M2_',
    'test_castlike_.*_expanded_',
    # others
    'test_maxpool_2d_ceil_',
    'test_averagepool_2d_ceil_',
    'test_averagepool_2d_dilations_',
    'test_nonzero_',
    # np.object is not valid type for jax.array
    'test_equal_string_',
])


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
