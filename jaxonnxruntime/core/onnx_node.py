# Copyright 2025 The Jaxonnxruntime Authors.
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
"""Wrap the onnx.NodeProto as OnnxNode class."""
import inspect
from typing import Any, Sequence
from jax import numpy as jnp
from jaxonnxruntime.core.onnx_utils import contain_subgraph
from jaxonnxruntime.core.onnx_utils import get_graph_input
import onnx


def convert_onnx(attr_proto: onnx.AttributeProto) -> Any:
  """Convert an ONNX attribute to a Python object.

  Args:
    attr_proto: An ONNX AttributeProto object.

  Returns:
    A Python object corresponding to the attribute value.

  Raises:
    ValueError: If the attribute type is not supported.
  """
  if attr_proto.HasField('f'):
    return attr_proto.f
  elif attr_proto.HasField('i'):
    return attr_proto.i
  elif attr_proto.HasField('s'):
    return str(attr_proto.s, 'utf-8')
  elif attr_proto.HasField('t'):
    return attr_proto.t  # this is a proto!
  elif attr_proto.HasField('g'):
    return attr_proto.g
  elif attr_proto.floats:
    return tuple(attr_proto.floats)
  elif attr_proto.ints:
    return tuple(attr_proto.ints)
  elif attr_proto.strings:
    str_list = tuple(map(lambda x: str(x, 'utf-8'), list(attr_proto.strings)))
    return str_list
  elif attr_proto.HasField('sparse_tensor'):
    return attr_proto.sparse_tensor
  else:
    raise ValueError('Unsupported ONNX attribute: {}'.format(attr_proto))


class OnnxNode:
  """A class that wraps an ONNX NodeProto as an OnnxNode object.

  Attributes:
    name (str): The name of the node.
    op_type (str): The type of the operation performed by the node.
    domain (str): The domain of the node.
    attrs (dict): A dictionary of attributes for the node, where the keys are
      the attribute names and the values are the attribute values.
    attrs_dict (dict): A dict of the attributes for the node, it is for the jax
      onnx implementation keyword arguments.
    inputs (list): A list of the node's input names.
    subgraph_inputs (list): A list of the input names of subgraphs.
    outputs (list): A list of the node's output names.
    node_proto (onnx.NodeProto): The underlying ONNX NodeProto object.
    context_graph (Any): The graph context that contains the node.
  """

  def __init__(self, node: onnx.NodeProto, context_graph: Any = None):
    """Creates an OnnxNode object from an ONNX NodeProto object.

    Args:
      node (onnx.NodeProto): The ONNX NodeProto object to wrap.
      context_graph (Any): The graph context that contains the node.
    """
    self.name: str = str(node.name)
    self.op_type: str = str(node.op_type)
    self.domain: str = str(node.domain)
    self.attrs: dict[str, Any] = dict(
        [(attr.name, convert_onnx(attr)) for attr in node.attribute]
    )
    self.attrs_dict: dict[str, Any] = {}
    self.inputs: list[str] = list(node.input)
    self.subgraph_inputs: list[str] = []
    self.outputs: list[str] = list(node.output)
    self.node_proto: onnx.NodeProto = node
    self.context_graph: Any = context_graph

    # For operators that involve control flow, OnnxNode is defined to be
    # a self-contained operator, different from Onnx.NodeProto.
    # The inputs to the subgraphs are added to the inputs to this parent
    # control flow operator.
    if contain_subgraph(node):
      for a in node.attribute:
        if a.HasField('g'):
          subg_inputs = get_graph_input(a.g)
          self.subgraph_inputs.extend(subg_inputs)

  @property
  def len_inputs(self) -> int:
    """The number of input tensors of the ONNX node."""
    return len(self.inputs)

  @property
  def len_outputs(self) -> int:
    """The number of output tensors of the ONNX node."""
    return len(self.outputs)

  def get_constant_node_value(self) -> Any:
    """Returns the value of the constant node."""
    assert self.node_proto.op_type == 'Constant', self.node_proto.op_type
    result = None
    attr_to_dtype = {
        'value_int': jnp.int64,
        'value_ints': jnp.int64,
        'value_float': jnp.float32,
        'value_floats': jnp.float32,
    }

    matched = 0
    if 'value_string' in self.attrs:
      result = self.attrs['value_string']
      matched = matched + 1
    elif 'value_strings' in self.attrs:
      result = self.attrs['value_strings']
      matched = matched + 1
    elif 'value' in self.attrs:
      result = onnx.numpy_helper.to_array(self.attrs['value'])
      matched = matched + 1
    else:
      for item in attr_to_dtype:
        if item in self.attrs:
          result = jnp.array(self.attrs[item], dtype=attr_to_dtype[item])
          matched = matched + 1

    assert (
        matched == 1
    ), f'Should only provide one of value attributes, but get {matched}'
    return result


def update_node_attr_dict_with_jax_func_kwargs(
    node: 'OnnxNode', onnx_jax_impl: Any
):
  """Update the node attributes dict with the jax onnx implementation kwargs."""
  sig = inspect.signature(onnx_jax_impl)
  kwparams = [
      param.name
      for param in sig.parameters.values()
      if param.kind == inspect.Parameter.KEYWORD_ONLY
  ]
  for name in kwparams:
    node.attrs_dict[name] = node.attrs.get(name, None)


def pad_sequence(sequence: Sequence[Any], length: int, pad_value: Any = None):
  """Pad a sequence to the length of the sequence."""
  assert len(sequence) <= length, f'{len(sequence)} >= {length}'
  return list(sequence) + [pad_value] * (length - len(sequence))
