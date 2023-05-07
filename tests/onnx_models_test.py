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
"""Test onnx real model."""
import collections
import unittest

import jax
from jaxonnxruntime.backend import Backend as JaxBackend
import onnx.backend.test
from onnx.backend.test.loader import load_model_tests

# Some node tests require jax_enable_x64=True.
# E.g. argmax, bitshift.
jax.config.update("jax_enable_x64", True)

# This is a pytest magic variable to load extra plugins
pytest_plugins = ("onnx.backend.test.report",)


class Runner(onnx.backend.test.runner.Runner):

  def __init__(self, backend, parent_module=None) -> None:
    self.backend = backend
    self._parent_module = parent_module
    self._include_patterns = set()
    self._exclude_patterns = set()
    self._xfail_patterns = set()
    self._test_items = collections.defaultdict(dict)

    for rt in load_model_tests(kind="real"):
      self._add_model_test(rt, "Real")

    for rt in load_model_tests(kind="simple"):
      self._add_model_test(rt, "Simple")


backend_test = Runner(JaxBackend, __name__)
expect_fail_patterns = []
include_patterns = []
exclude_patterns = []

include_patterns.append("test_resnet50_")

for pattern in include_patterns:
  backend_test.include(pattern)

for pattern in exclude_patterns:
  backend_test.exclude(pattern)

for pattern in expect_fail_patterns:
  backend_test.xfail(pattern)


class ModelTest(unittest.TestCase):
  pass


for name, func in backend_test.test_cases.items():
  setattr(ModelTest, name, func)


if __name__ == "__main__":
  unittest.main()
