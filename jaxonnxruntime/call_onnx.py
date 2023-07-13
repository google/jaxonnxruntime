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
from typing import Any, Callable, Dict, Sequence, Tuple, Type, Union

import jax
from jaxonnxruntime import config
from jaxonnxruntime import onnx_ops  # pylint: disable=unused-import
from jaxonnxruntime.core import handler as onnx_handler
from jaxonnxruntime.core import onnx_graph
from jaxonnxruntime.core import onnx_node
from jaxonnxruntime.core import onnx_utils

import onnx
from onnx import defs
from onnx.helper import make_opsetid


OnnxNode = onnx_node.OnnxNode
OnnxGraph = onnx_graph.OnnxGraph
Handler = onnx_handler.Handler

logger = logging.getLogger(__name__)


def call_onnx_model(
    model: onnx.ModelProto,
    inputs: Union[Sequence[Any], Dict[str, Any]],
    rename_tensors: bool = False,
) -> Tuple[Callable[..., Any], Any]:
  """Convert an ONNX.ModelProto to a JAX function with model parameters and sample input.

  Args:
    model: The ONNX model to convert.
    inputs: The sample input(s) for the model. It can be either a sequence of
      inputs or a dictionary mapping input names to values.
    rename_tensors: Indicates whether to rename all onnx.TensorProto name with
      unique id `tensor_{id}`. Default is False.

  Returns:
    model_func, model_params: A tuple containing the JAX function  and the
    model_params as PyTree.
  """

  graph = model.graph
  if rename_tensors:
    graph = onnx_utils.sanitize_tensor_names_in_graph(graph)
  if model.ir_version < 3:
    opset = [make_opsetid(defs.ONNX_DOMAIN, 1)]
  else:
    opset = model.opset_import
  model_params = {
      n.name: onnx_utils.valueinfoproto_asarray(n) for n in graph.initializer
  }
  input_names = onnx_utils.get_graph_input(graph)
  tensor_dict = dict(
      **onnx_utils.maybe_convert_to_dict(inputs, input_names), **model_params
  )
  model_func = call_onnx_graph(graph, tensor_dict, opset=opset)
  del tensor_dict
  return model_func, model_params


def call_onnx_graph(
    graph: onnx.GraphProto,
    tensor_dict: Dict[str, Any],
    opset: ... = None,
) -> Callable[..., Any]:
  """Convert ONNX.GraphProto to jax_func with ONNX.GraphProto.initializer as parameters."""
  tensor_ref_dict = build_ref_dict(graph)
  graph_helper = OnnxGraph(graph)

  # step 1: Trace those static info
  jit_func_dict = {}
  onnx_node_dict = {}
  if opset is None:
    opset = [make_opsetid(defs.ONNX_DOMAIN, defs.onnx_opset_version())]
  handlers = _get_all_handlers(opset)
  node_execute_order_list = graph_helper.topological_sort()

  logger.info('Start tracing the jax_func model to get some static info')
  for node_proto in node_execute_order_list:
    node = OnnxNode(node_proto, graph_helper)
    onnx_node_dict[node.name] = node
    try:
      node_inputs = [tensor_dict[x] for x in node.inputs + node.subgraph_inputs]
    except Exception as e:
      raise ValueError(
          'Fail to get the input tensor of node input names'
          f'{node.inputs + node.subgraph_inputs}, the node proto is'
          f'{node.node_proto}.'
      ) from e
    jit_func = _get_jit_func(node, node_inputs, handlers=handlers)
    jit_func_dict[node.name] = jit_func

    if config.jaxort_experimental_support_abtract_input_shape:
      outputs = jax.eval_shape(jit_func, *node_inputs, **node.attrs_dict)
    else:
      outputs = jit_func(*node_inputs, **node.attrs_dict)
    outputs = outputs if isinstance(outputs, Sequence) else [outputs]

    for name, output in zip(node.outputs, outputs):
      tensor_dict[name] = output

  input_names = onnx_utils.get_graph_input(graph)

  def model_func(model_params, inputs):
    tensor_dict = dict(
        **onnx_utils.maybe_convert_to_dict(inputs, input_names), **model_params
    )
    ref_dict = {}
    for node_proto in node_execute_order_list:
      node = onnx_node_dict[node_proto.name]
      node_inputs = [tensor_dict[x] for x in node.inputs + node.subgraph_inputs]
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

      for input_ in node.inputs + node.subgraph_inputs:
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

  return model_func


def build_ref_dict(graph: onnx.GraphProto) -> Dict[str, int]:
  """Initialize reference count dict."""
  ref_dict: dict[Any, Any] = {}
  for node in graph.node:
    if onnx_utils.contain_subgraph(node):
      for a in node.attribute:
        if a.HasField('g'):
          sub_ref_dict = build_ref_dict(a.g)
          ref_dict.update(
              {k: ref_dict.get(k, 0) + v for k, v in sub_ref_dict.items()}
          )
    inputs = node.input
    for input_ in inputs:
      if input_ in ref_dict:
        ref_dict[input_] += 1
      else:
        ref_dict[input_] = 1
  for o in graph.output:
    ref_dict[o.name] = ref_dict[o.name] + 1 if o.name in ref_dict else 1
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
    if handler.DOMAIN not in opset_dict:
      raise ValueError(
          f'handler.DOMAIN {handler.DOMAIN} is not in opset_dict {opset_dict}'
      )
    version = opset_dict[handler.DOMAIN]
    since_version = handler.get_since_version(version)
    handler.SINCE_VERSION = since_version
    handlers.setdefault(domain, {})[handler.OP_TYPE] = handler

  return handlers


def _get_jit_func(
    node: OnnxNode,
    inputs: list[Any],
    handlers: Dict[str, Dict[str, type[Handler]]],
    **kwargs,
):
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
