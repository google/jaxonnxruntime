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

"""onnx utility functions collection."""

import inspect
from typing import Any, Dict, Optional, Sequence, Union

from absl import logging
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from jaxonnxruntime import config as jort_config
from jaxonnxruntime.core import onnx_graph
import numpy as np

import onnx
from onnx import numpy_helper


def tensor_dtype_to_jnp_dtype(
    tensor_type: onnx.TensorProto.DataType,
) -> jnp.dtype:
  """Convert onnx.TensorProto.DataType to jnp.dtype."""
  if tensor_type is onnx.TensorProto.BFLOAT16:
    return jnp.bfloat16
  if onnx.__version__ < "1.14.0":
    np_type = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[tensor_type]
  else:
    np_type = onnx.helper.tensor_dtype_to_np_dtype(tensor_type)
  return jnp.dtype(np_type)


def get_elem_type_from_type_proto(type_proto: onnx.TypeProto):
  if type_proto.HasField("optional_type"):
    return get_elem_type_from_type_proto(type_proto.optional_type.elem_type)
  if type_proto.HasField("sequence_type"):
    return get_elem_type_from_type_proto(type_proto.sequence_type.elem_type)

  if type_proto.HasField("tensor_type"):
    return type_proto.tensor_type.elem_type

  raise ValueError(
      f"currently only support Tensor type TypeProto but got {type_proto}"
  )


def get_shape_and_dtype_from_val_info(
    value_info: onnx.ValueInfoProto,
) -> tuple[list[int], jnp.dtype]:
  """Get jax numpy shape and dtype from onnx.ValueInfoProto."""
  type_proto = value_info.type
  elem_type = get_elem_type_from_type_proto(type_proto)
  dtype = tensor_dtype_to_jnp_dtype(elem_type)
  shape = [dim.dim_value for dim in type_proto.tensor_type.shape.dim]

  return shape, dtype


def contain_subgraph(node: Any) -> bool:
  """Check if the node contains subgraph (control flow)."""
  return node.op_type in ("If", "Loop")


def get_graph_input(graph: onnx.GraphProto) -> list[str]:
  """Returns unique non-node input names."""
  exclude_set: set[str] = set(ts.name for ts in graph.initializer)
  real_input: list[str] = [
      ts.name for ts in graph.input if ts.name not in exclude_set
  ]
  exclude_set.update(real_input)
  for node in graph.node:
    exclude_set.update(name for name in node.output)
  for node in graph.node:
    real_input.extend(i for i in node.input if i not in exclude_set)

  # Sometimes input name is empty string(""), which should be removed.
  # We also need to remove duplicates.
  unique_real_input = []
  for item in real_input:
    if item not in unique_real_input and item != "":  # pylint: disable=g-explicit-bool-comparison
      unique_real_input.append(item)
  return unique_real_input


def valueinfoproto_asarray(proto: Any) -> jax.Array:
  """Convert onnx.ValueInfoProto to jaxlib.xla_extension.ArrayImpl."""
  return jnp.asarray(numpy_helper.to_array(proto).reshape(tuple(proto.dims)))


def maybe_convert_to_dict(
    inputs: Union[Sequence[Any], Dict[str, Any]],
    input_names: Optional[Sequence[Any]] = None,
):
  """Convert inputs to a dictionary with input_names as keys."""
  if isinstance(inputs, dict):
    return inputs
  elif isinstance(inputs, Sequence):
    if input_names is None:
      raise ValueError("Should provide input names if `inputs` is a Sequence!")
    assert len(inputs) == len(input_names)
    return dict(zip(input_names, inputs))
  else:
    raise NotImplementedError("Please use inputs of type dict or Sequence!")


def sanitize_tensor_names_in_graph(
    graph: onnx.GraphProto,
) -> onnx.GraphProto:
  """Format the names of all tensors in an onnx.GraphProto.

  Each tensors will have a unique name in the format 'tensor_{idx}'.
  Args:
    graph: the onnx.GraphProto to be processed.

  Returns:
    graph: the graph within which tensor names have been formatted.
  """

  def _unique_tensor_name_generator():
    idx = 0
    while True:
      yield f"tensor_{str(idx)}"
      idx += 1

  unique_name_gen = _unique_tensor_name_generator()
  name_map = {}

  def _sanitize_tensor_names_in_graph(graph):
    for nd in graph.node:
      for i in range(len(nd.input)):
        if nd.input[i] not in name_map:
          name_map[nd.input[i]] = next(unique_name_gen)
        nd.input[i] = name_map[nd.input[i]]
      for i in range(len(nd.output)):
        if nd.output[i] not in name_map:
          name_map[nd.output[i]] = next(unique_name_gen)
        nd.output[i] = name_map[nd.output[i]]
      if contain_subgraph(nd):
        for attr_proto in nd.attribute:
          if attr_proto.HasField("g"):
            _sanitize_tensor_names_in_graph(attr_proto.g)
    for proto in graph.initializer:
      if proto.name not in name_map:
        name_map[proto.name] = next(unique_name_gen)
      proto.name = name_map[proto.name]
    for proto in graph.input:
      if proto.name not in name_map:
        name_map[proto.name] = next(unique_name_gen)
      proto.name = name_map[proto.name]
    for proto in graph.output:
      if proto.name not in name_map:
        name_map[proto.name] = next(unique_name_gen)
      proto.name = name_map[proto.name]

  _sanitize_tensor_names_in_graph(graph)
  return graph


def with_jax_config(**kwds):
  """Test case decorator for subclasses of JortTestCase."""

  def decorator(cls):
    assert inspect.isclass(cls) and issubclass(
        cls, JortTestCase
    ), "@with_jax_config can only wrap JaxTestCase class definitions."
    cls.default_jax_config = {**JortTestCase.default_jax_config, **kwds}
    return cls

  return decorator


def with_jort_config(**kwds):
  """Test case decorator for subclasses of JortTestCase."""

  def decorator(cls):
    assert inspect.isclass(cls) and issubclass(
        cls, JortTestCase
    ), "@with_jax_config can only wrap JaxTestCase class definitions."
    cls.default_jort_config = {**JortTestCase.default_jort_config, **kwds}
    return cls

  return decorator


def is_sequence(x):
  try:
    iter(x)
  except TypeError:
    return False
  else:
    return True


def _cosin_sim(a: Any, b: Any) -> float:
  a = np.array(a)
  b = np.array(b)
  a = a.astype(jnp.float32)
  b = b.astype(jnp.float32)
  a = a.flatten()
  b = b.flatten()
  cos_sim = jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b))
  return cos_sim


class JortTestCase(parameterized.TestCase):
  """Base class for JAXOnnxRuntime tests."""

  default_jort_config = {}
  default_jax_config = {}

  def setUp(self):
    """Set the jax and jaxonnxruntime config."""
    super().setUp()

    self._original_jax_config = {}
    for key, value in self.default_jax_config.items():
      self._original_jax_config[key] = jax.config._read(key)
      jax.config.update(key, value)

    self._original_jort_config = {}
    for key, value in self.default_jort_config.items():
      self._original_jax_config[key] = jort_config._read(key)
      jort_config.update(key, value)

  def tearDown(self):
    """Reset the jax and jaxonnxruntime config."""
    for key, value in self._original_jax_config.items():
      jax.config.update(key, value)
    for key, value in self._original_jort_config.items():
      jort_config.update(key, value)
    super().tearDown()

  def assert_allclose(self, x, y, *, atol=10e-7, rtol=10e-5, err_msg=""):
    """Assert that x and y, either arrays or nested tuples/lists, are close."""
    if isinstance(x, dict):
      self.assertIsInstance(y, dict)
      self.assertEqual(set(x.keys()), set(y.keys()))
      for k in x.keys():
        self.assert_allclose(x[k], y[k], atol=atol, rtol=rtol, err_msg=err_msg)
    elif is_sequence(x) and not hasattr(x, "__array__"):
      self.assertTrue(is_sequence(y) and not hasattr(y, "__array__"))
      self.assertEqual(len(x), len(y))
      for x_elt, y_elt in zip(x, y):
        self.assert_allclose(
            x_elt, y_elt, atol=atol, rtol=rtol, err_msg=err_msg
        )
    elif hasattr(x, "__array__") or np.isscalar(x):
      self.assertTrue(hasattr(y, "__array__") or np.isscalar(y), type(y))
      x = np.asarray(x)
      y = np.asarray(y)
      np.testing.assert_allclose(x, y, atol=atol, rtol=rtol, err_msg=err_msg)
    elif x == y:
      return
    else:
      raise TypeError((type(x), type(y)))

  def assert_all_similar(self, x, y, *, similarity=0.97, err_msg=""):
    """Assert that x and y, either arrays or nested tuples/lists, are have linear similarity."""
    if isinstance(x, dict):
      self.assertIsInstance(y, dict)
      self.assertEqual(set(x.keys()), set(y.keys()))
      for k in x.keys():
        self.assert_all_similar(
            x[k], y[k], similarity=similarity, err_msg=err_msg
        )
    elif is_sequence(x) and not hasattr(x, "__array__"):
      self.assertTrue(is_sequence(y) and not hasattr(y, "__array__"))
      self.assertEqual(len(x), len(y))
      for x_elt, y_elt in zip(x, y):
        self.assert_all_similar(
            x_elt, y_elt, similarity=similarity, err_msg=err_msg
        )
    elif hasattr(x, "__array__") or np.isscalar(x):
      self.assertTrue(hasattr(y, "__array__") or np.isscalar(y), type(y))
      x = np.asarray(x)
      y = np.asarray(y)
      self.assertGreater(_cosin_sim(x, y), similarity, err_msg)
    elif x == y:
      return
    else:
      raise TypeError((type(x), type(y)))

  def assert_ort_jort_all_close(
      self, onnx_model: onnx.ModelProto, model_inputs: tuple[Any]
  ):
    """Assert ONNXRuntime and JaxOnnxRuntime is numerically close."""
    # Update the onnx_model with all intermediate outputs.
    graph_helper = onnx_graph.OnnxGraph(onnx_model.graph)
    node_execute_order_list = graph_helper.topological_sort()
    all_outputs = []
    for node in node_execute_order_list:
      all_outputs.extend(node.output)

    all_outputs = list(
        filter(lambda n: n in graph_helper.value_info_dict, all_outputs)
    )

    all_outputs_value_info = [
        graph_helper.value_info_dict[name] for name in all_outputs
    ]

    while onnx_model.graph.output:
      onnx_model.graph.output.pop()
    onnx_model.graph.output.extend(all_outputs_value_info)

    from jaxonnxruntime import backend as jort_backend  #  pylint: disable=g-import-not-at-top

    try:
      import onnxruntime.backend as ort_backend  #  pylint: disable=g-import-not-at-top
    except ImportError:
      ort_backend = None

    if ort_backend:
      ort_model = ort_backend.prepare(onnx_model)
      result_ort = ort_backend.run(ort_model, model_inputs)
      result_jort = jort_backend.run(onnx_model, model_inputs)
      assert len(result_ort) == len(
          result_jort
      ), f"{len(result_ort)} != {len(result_jort)}"
      assert len(result_jort) == len(all_outputs)
      for i in range(len(all_outputs)):
        self.assert_all_similar(
            result_ort[i],
            result_jort[i],
            err_msg=(
                f"Tensor {all_outputs[i]} jort and ort mismatch:\n"
                f"jort={result_jort[i]}, ort={result_ort[i]}"
            ),
        )
    else:
      logging.info("Please install onnxruntime first.")

  def assert_model_run_through(
      self, onnx_model: onnx.ModelProto, model_inputs: tuple[Any]
  ):
    from jaxonnxruntime import backend as jort_backend  #  pylint: disable=g-import-not-at-top

    prepared_model = jort_backend.prepare(onnx_model)
    assert prepared_model is not None
    _ = prepared_model.run(model_inputs)
