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

"""ONNX OP test runner."""

import collections
from collections.abc import Callable, Iterable, Sequence
import dataclasses
import functools
import glob
import json
import os
import re
import sys
import time
from typing import Any, Optional, Pattern, Set, Type, Union
import unittest

import jax
from jaxonnxruntime.backend import Backend
import numpy as np

import onnx
from onnx import numpy_helper


jax.config.update('jax_enable_x64', True)
jax.config.update('jax_numpy_rank_promotion', 'warn')


class TestItem:

  def __init__(
      self,
      func: Callable[..., Any],
      proto: list[Optional[Union[onnx.ModelProto, onnx.NodeProto]]],
  ) -> None:
    self.func = func
    self.proto = proto


@dataclasses.dataclass
class TestCase:
  """A dataclass representing a test case."""

  name: str
  model_name: str
  url: Optional[str]
  model_dir: Optional[str]
  model: Optional[onnx.ModelProto]
  data_sets: Optional[
      Sequence[tuple[Sequence[np.ndarray], Sequence[np.ndarray]]]
  ]
  kind: str
  rtol: float
  atol: float
  # Tell PyTest this isn't a real test.
  __test__: bool = False


DATA_DIR = os.path.join(
    os.path.realpath(os.path.dirname(onnx.__file__)), 'backend/test/data'
)


class BackendIsNotSupposedToImplementIt(unittest.SkipTest):  # pylint: disable=g-bad-exception-name
  """Raised when the backend is not supposed to implement."""


def retry_execute(
    times: int,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
  """Decorator that retries executing the decorated function a specified number of times."""
  assert times >= 1

  def wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
      for i in range(1, times + 1):
        try:
          return func(*args, **kwargs)
        except Exception:  # pylint: disable=broad-except
          print(f'{i} times tried')
          if i == times:
            raise
          time.sleep(5 * i)

    return wrapped

  return wrapper


class Runner:
  """Unit test runner."""

  @classmethod
  def load_model_tests(
      cls,
      data_dir: str = DATA_DIR,
      kind: Optional[str] = None,
  ) -> list[TestCase]:
    """Load model test cases from on-disk data files."""

    supported_kinds = os.listdir(data_dir)
    if kind not in supported_kinds:
      raise ValueError(f'kind must be one of {supported_kinds}')

    testcases = []

    kind_dir = os.path.join(data_dir, kind)
    for test_name in os.listdir(kind_dir):
      case_dir = os.path.join(kind_dir, test_name)
      # skip the non-dir files, such as generated __init__.py.
      rtol = 1e-3
      atol = 1e-7
      if not os.path.isdir(case_dir):
        continue
      if os.path.exists(os.path.join(case_dir, 'model.onnx')):
        url = None
        model_name = test_name[len('test_')]
        model_dir: Optional[str] = case_dir
      else:
        with open(os.path.join(case_dir, 'data.json')) as f:
          data = json.load(f)
          url = data['url']
          model_name = data['model_name']
          rtol = data.get('rtol', 1e-3)
          atol = data.get('atol', 1e-7)
          model_dir = None
      testcases.append(
          TestCase(
              name=test_name,
              url=url,
              model_name=model_name,
              model_dir=model_dir,
              model=None,
              data_sets=None,
              kind=kind,
              rtol=rtol,
              atol=atol,
          )
      )
    return testcases

  def __init__(
      self, backend: Type[Backend], parent_module: Optional[str] = None
  ) -> None:
    self.backend = backend
    self._parent_module = parent_module
    self._include_patterns: Set[Pattern[str]] = set()
    self._exclude_patterns: Set[Pattern[str]] = set()
    self._xfail_patterns: Set[Pattern[str]] = set()

    # This is the source of the truth of all test functions.
    # {category: {name: func}}
    self._test_items: dict[str, dict[str, TestItem]] = collections.defaultdict(
        dict
    )

  def _get_test_case(self, name: str) -> Type[unittest.TestCase]:
    test_case = type(str(name), (unittest.TestCase,), {})
    if self._parent_module:
      test_case.__module__ = self._parent_module
    return test_case

  def include(self, pattern: str):
    self._include_patterns.add(re.compile(pattern))
    return self

  def exclude(self, pattern: str):
    self._exclude_patterns.add(re.compile(pattern))
    return self

  def xfail(self, pattern: str):
    self._xfail_patterns.add(re.compile(pattern))
    return self

  @property
  def _filtered_test_items(self) -> dict[str, dict[str, TestItem]]:
    """Property that returns the filtered test items based on the include, exclude, and xfail patterns."""

    filtered: dict[str, dict[str, TestItem]] = {}
    for category, items_map in self._test_items.items():
      filtered[category] = {}
      for name, item in items_map.items():
        if self._include_patterns and (
            not any(include.search(name) for include in self._include_patterns)
        ):
          item.func = unittest.skip('no matched include pattern')(item.func)
        for exclude in self._exclude_patterns:
          if exclude.search(name):
            item.func = unittest.skip(
                f'matched exclude pattern "{exclude.pattern}"'
            )(item.func)
        for xfail in self._xfail_patterns:
          if xfail.search(name):
            item.func = unittest.expectedFailure(item.func)
        filtered[category][name] = item
    return filtered

  @property
  def test_cases(self) -> dict[str, Type[unittest.TestCase]]:
    """List of test cases to be applied on the parent scope.

    Example usage:
        globals().update(BackendTest(backend).test_cases)
    """
    test_cases = {}
    for category, items_map in self._filtered_test_items.items():
      test_case_name = f'OnnxBackend{category}Test'
      test_case = self._get_test_case(test_case_name)
      for name, item in sorted(items_map.items()):
        setattr(test_case, name, item.func)
      test_cases[test_case_name] = test_case
    return test_cases

  @property
  def test_suite(self) -> unittest.TestSuite:
    """TestSuite that can be run by TestRunner.

    Example usage:
        unittest.TextTestRunner().run(BackendTest(backend).test_suite)
    """
    suite = unittest.TestSuite()
    for case in sorted(self.test_cases.values()):  # type: ignore
      suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(case))
    return suite

  @classmethod
  def assert_similar_outputs(
      cls,
      ref_outputs: Sequence[Any],
      outputs: Sequence[Any],
      rtol: float,
      atol: float,
  ) -> None:
    """Assert that two sequences of outputs are similar within given tolerances."""
    np.testing.assert_equal(len(outputs), len(ref_outputs))
    for i, _ in enumerate(outputs):
      if isinstance(outputs[i], (list, tuple)):
        for j, _ in enumerate(outputs[i]):
          cls.assert_similar_outputs(
              ref_outputs[i][j], outputs[i][j], rtol, atol
          )
      else:
        np.testing.assert_equal(outputs[i].dtype, ref_outputs[i].dtype)
        if ref_outputs[i].dtype == object:
          np.testing.assert_array_equal(outputs[i], ref_outputs[i])
        else:
          np.testing.assert_allclose(
              outputs[i], ref_outputs[i], rtol=rtol, atol=atol
          )

  def _add_test(
      self,
      category: str,
      test_name: str,
      test_func: Callable[..., Any],
      report_item: list[Optional[Union[onnx.ModelProto, onnx.NodeProto]]],
      devices: Iterable[str] = ('CPU', 'CUDA'),
  ) -> None:
    """Add test to each device and category."""
    # We don't prepend the 'test_' prefix to improve greppability
    if not test_name.startswith('test_'):
      raise ValueError(f'Test name must start with test_: {test_name}')

    def add_device_test(device: str) -> None:
      device_test_name = f'{test_name}_{device.lower()}'
      if device_test_name in self._test_items[category]:
        raise ValueError(
            'Duplicated test name "{}" in category "{}"'.format(
                device_test_name, category
            )
        )

      @unittest.skipIf(  # type: ignore
          not self.backend.supports_device(device),
          f"Backend doesn't support device {device}",
      )
      @functools.wraps(test_func)
      def device_test_func(*args: Any, **kwargs: Any) -> Any:
        try:
          return test_func(*args, device=device, **kwargs)
        except BackendIsNotSupposedToImplementIt as e:
          # hacky verbose reporting
          if '-v' in sys.argv or '--verbose' in sys.argv:
            print(
                'Test {} is effectively skipped: {}'.format(device_test_name, e)
            )

      self._test_items[category][device_test_name] = TestItem(
          device_test_func, report_item
      )

    for device in devices:
      add_device_test(device)

  def _add_model_test(self, model_test, kind: str) -> None:
    """model is loaded at runtime, note sometimes it could even never loaded if the test skipped."""
    model_marker: list[Optional[Union[onnx.ModelProto, onnx.NodeProto]]] = [
        None
    ]

    def run(test_self: Any, device: str) -> None:  # pylint: disable=unused-argument
      model_dir = model_test.model_dir
      model_pb_path = os.path.join(model_dir, 'model.onnx')
      model = onnx.load(model_pb_path)
      model_marker[0] = model
      if (
          hasattr(self.backend, 'is_compatible')
          and callable(self.backend.is_compatible)
          and not self.backend.is_compatible(model, device)
      ):
        raise unittest.SkipTest('Not compatible with backend')
      prepared_model = self.backend.prepare(model, device)
      assert prepared_model is not None

      for test_data_npz in glob.glob(
          os.path.join(model_dir, 'test_data_*.npz')
      ):
        test_data = np.load(test_data_npz, encoding='bytes')
        inputs = list(test_data['inputs'])
        outputs = list(prepared_model.run(inputs))
        ref_outputs = test_data['outputs']
        self.assert_similar_outputs(
            ref_outputs, outputs, rtol=model_test.rtol, atol=model_test.atol
        )
      for test_data_dir in glob.glob(os.path.join(model_dir, 'test_data_set*')):
        inputs = []
        inputs_num = len(glob.glob(os.path.join(test_data_dir, 'input_*.pb')))
        for i in range(inputs_num):
          input_file = os.path.join(test_data_dir, f'input_{i}.pb')
          self._load_proto(input_file, inputs, model.graph.input[i].type)
        ref_outputs = []
        ref_outputs_num = len(
            glob.glob(os.path.join(test_data_dir, 'output_*.pb'))
        )
        for i in range(ref_outputs_num):
          output_file = os.path.join(test_data_dir, f'output_{i}.pb')
          self._load_proto(output_file, ref_outputs, model.graph.output[i].type)
        outputs = list(prepared_model.run(inputs))
        self.assert_similar_outputs(
            ref_outputs, outputs, rtol=model_test.rtol, atol=model_test.atol
        )

    self._add_test(kind + 'Model', model_test.name, run, model_marker)

  def _load_proto(
      self,
      proto_filename: str,
      target_list: list[Union[np.ndarray, list[Any]]],
      model_type_proto,
  ) -> None:
    """Load protobuf file and add it into target_list."""
    with open(proto_filename, 'rb') as f:
      protobuf_content = f.read()
      if model_type_proto.HasField('sequence_type'):
        sequence = onnx.SequenceProto()
        sequence.ParseFromString(protobuf_content)
        target_list.append(numpy_helper.to_list(sequence))
      elif model_type_proto.HasField('tensor_type'):
        tensor = onnx.TensorProto()
        tensor.ParseFromString(protobuf_content)
        target_list.append(numpy_helper.to_array(tensor))
      elif model_type_proto.HasField('optional_type'):
        optional = onnx.OptionalProto()
        optional.ParseFromString(protobuf_content)
        target_list.append(numpy_helper.to_optional(optional))  # type: ignore
      else:
        print(
            'Loading proto of that specific type (Map/Sparse Tensor) is'
            ' currently not supported'
        )


load_model_tests = Runner.load_model_tests
