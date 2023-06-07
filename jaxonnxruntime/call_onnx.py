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
"""Convert ONNX model into jax function."""

import logging
from typing import Any, Callable, Dict, Sequence, Type, Tuple, Union

from jax import numpy as jnp

from jaxonnxruntime import onnx_ops  # pylint: disable=unused-import
from jaxonnxruntime.core import handler as onnx_handler
from jaxonnxruntime.core import onnx_graph
from jaxonnxruntime.core import onnx_node
import onnx
from onnx import defs
from onnx import numpy_helper
from onnx.helper import make_opsetid

OnnxNode = onnx_node.OnnxNode
OnnxGraph = onnx_graph.OnnxGraph
Handler = onnx_handler.Handler

logger = logging.getLogger(__name__)


def call_onnx(
    model: onnx.ModelProto, inputs: Union[Sequence[Any], Dict[str, Any]]
) -> Tuple[Callable[..., Any], Any]:
  """Convert. ONNX model to jax_func with model parameters."""

  graph = model.graph
  model_func, model_params = construct_jax_modules(graph, inputs)

  return model_func, model_params


def contain_subgraph(node):
  return node.op_type in ('If', 'Loop')


def flatten_subgraph(node, tensor_dict):
  for a_name, a in node.attrs.items():
    if isinstance(a, onnx.GraphProto):
      input_names = a.input
      inputs = {k: tensor_dict[k] for k in input_names}
      jax_func, params = construct_jax_modules(a, inputs)
      node.attrs_dict[a_name] = (jax_func, params)
      node.attrs_dict[f'{a_name}_input_num'] = len(input_names)
      node.inputs.extend(input_names)


def construct_jax_modules(
    graph: onnx.GraphProto, inputs: Union[Sequence[Any], Dict[str, Any]]
) -> Tuple[Callable[..., Any], Any]:
  """Convert ONNX GraphProto to jax_func with model parameters."""
  tensor_ref_dict = build_ref_dict(graph)
  graph_helper = OnnxGraph(graph)

  # step 1: Trace those static info
  def _asarray(proto):
    return jnp.asarray(numpy_helper.to_array(proto).reshape(tuple(proto.dims)))

  model_params = {n.name: _asarray(n) for n in graph.initializer}
  jit_func_dict = {}
  onnx_node_dict = {}
  opset = [make_opsetid(defs.ONNX_DOMAIN, defs.onnx_opset_version())]
  handlers = _get_all_handlers(opset)
  node_execute_order_list = graph_helper.topological_sort()

  def _maybe_convert_to_dict(inputs):
    if isinstance(inputs, dict):
      return inputs
    elif isinstance(inputs, Sequence):
      graph_inputs = graph_helper.get_real_input()
      assert len(inputs) == len(graph_inputs)
      return dict(zip(graph_inputs, inputs))
    else:
      raise NotImplementedError('Please use inputs of type dict or Sequence!')

  tensor_dict = dict(**_maybe_convert_to_dict(inputs), **model_params)

  logger.info('Start tracing the jax_func model to get some static info')
  for node_proto in node_execute_order_list:
    node = OnnxNode(node_proto, graph_helper)
    onnx_node_dict[node.name] = node
    if contain_subgraph(node):
      flatten_subgraph(node, tensor_dict)
    try:
      node_inputs = [tensor_dict[x] for x in node.inputs]
    except Exception as e:
      raise ValueError(
          f'failt to get input tensor for node inputs {node.inputs}, the'
          f' node proto is {node.node_proto}'
      ) from e
    jit_func = _get_jit_func(node, node_inputs, handlers=handlers)
    jit_func_dict[node.name] = jit_func
    outputs = jit_func(*node_inputs, **node.attrs_dict)
    outputs = outputs if isinstance(outputs, Sequence) else [outputs]

    for name, output in zip(node.outputs, outputs):
      tensor_dict[name] = output
  del tensor_dict

  def model_func(model_params, inputs):
    tensor_dict = dict(**_maybe_convert_to_dict(inputs), **model_params)
    ref_dict = {}
    for node_proto in node_execute_order_list:
      node = onnx_node_dict[node_proto.name]
      node_inputs = [tensor_dict[x] for x in node.inputs]
      jit_func = jit_func_dict[node.name]
      outputs = jit_func(*node_inputs, **node.attrs_dict)
      outputs = outputs if isinstance(outputs, Sequence) else [outputs]

      for name, output in zip(node.outputs, outputs):
        tensor_dict[name] = output

      # Below lines may cause error when tensor_dict[x] is a list
      # Comment out because they are only for debugging
      # node_input_shapes = [tensor_dict[x].shape for x in node.inputs]
      # node_output_shapes = [tensor_dict[x].shape for x in node.outputs]
      # logger.debug('\t%s  -> %s', node_input_shapes, node_output_shapes)

      for input_ in node.inputs:
        if input_ in ref_dict:
          ref_dict[input_] += 1
        else:
          ref_dict[input_] = 1
      remove_keys = []
      for k, v in ref_dict.items():
        if tensor_ref_dict[k] == v:
          remove_keys.append(k)
      for rm_k in remove_keys:
        del ref_dict[rm_k]
        del tensor_dict[rm_k]

    return [tensor_dict[n.name] for n in graph.output]

  return model_func, model_params


def build_ref_dict(graph: onnx.GraphProto) -> Dict[str, int]:
  """Initialize reference count dict."""
  ref_dict: dict[Any, Any] = {}
  for node in graph.node:
    inputs = node.input
    for input_ in inputs:
      if input_ in ref_dict:
        ref_dict[input_] += 1
      else:
        ref_dict[input_] = 1

  return ref_dict


def _get_all_handlers(
    opset: Sequence[onnx.OperatorSetIdProto],
) -> Dict[str, Dict[str, Type[Handler]]]:
  """Get all ONNX OP_TYPE handlers from Handler subclasses.

  Args:
      opset: An OperatorSetIdProto message containing the operator set version
        information.

  Returns:
      A dictionary of all the ONNX handlers, where the keys are the domain
      names
      and the values are nested dictionaries mapping operator names to their
      Handler
      subclasses.

  Raises:
      ValueError: If there is no OP_TYPE attribute defined in the Handler class.
  """

  handlers: Dict[Any, Any] = {}
  for handler in Handler.__subclasses__():
    if not hasattr(handler, 'OP_TYPE'):
      logger.warning(
          (
              "%s doesn't have ONNX OP_TYPE. "
              'Please use handler.register_op decorator to register it.'
          ),
          handler.__name__,
      )

    domain = handler.DOMAIN
    opset_dict = dict([(o.domain, o.version) for o in opset])
    version = opset_dict[handler.DOMAIN]
    since_version = handler.get_since_version(version)
    handler.SINCE_VERSION = since_version
    handlers.setdefault(domain, {})[handler.OP_TYPE] = handler

  return handlers


def _get_jit_func(node, inputs, handlers, **kwargs):
  """Get the JAX node implementation."""
  handler = (
      handlers[node.domain].get(node.op_type, None)
      if node.domain in handlers
      else None
  )
  if handler:
    return handler.handle(node, inputs, **kwargs)
  else:
    raise NotImplementedError(f'{node.op_type} is not implemented.')
