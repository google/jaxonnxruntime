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

"""Quick example how to run resnet50 with jaxonnxruntime."""

# Users can do more experiments based on flax resnet50 examples
# https://github.com/google/flax/tree/main/examples/imagenet

import logging

from absl import app
from absl import flags
import jax
from jax import numpy as jnp
from jaxonnxruntime import backend as JaxBackend
from jaxonnxruntime import call_onnx
import numpy as np
from orbax.export import ExportManager
from orbax.export import JaxModule
from orbax.export import ServingConfig
import tensorflow as tf


import onnx
from onnx import hub


FLAGS = flags.FLAGS
flags.DEFINE_string(
    "model_name",
    None,
    "Model name from ONNX Hub, for example: resnet50. See"
    " onnx.hub.list_models() to find full list.",
    required=True,
)

flags.DEFINE_string(
    "output_dir", None, "Diretory to export the model.", required=True
)


def _download_model(model_name):
  try:
    logging.info("Start downloading model %s", model_name)
    model = hub.load(model_name)
  except Exception as e:
    raise RuntimeError(f"Fail to download model {model_name}.") from e

  return model


def _cosin_sim(a, b):
  a = np.array(a)
  b = np.array(b)
  a = a.astype(jnp.float32)
  b = b.astype(jnp.float32)
  a = a.flatten()
  b = b.flatten()
  cos_sim = jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b))
  return cos_sim


def run_model_from_onnx_backend():
  """Test standard ONNX Backend API on onnx.hub models."""

  model_name_list = list(map(lambda x: x.model.lower(), hub.list_models()))
  model_name = FLAGS.model_name
  assert (
      model_name in model_name_list
  ), f"{model_name} is valid model name in onnx.hub."
  model = _download_model(model_name)

  try:
    onnx.checker.check_model(model)
  except AttributeError as e:
    logging.warning("Fail to load onnx.checker, skip the check. %s", e)

  model_info = hub.get_model_info(model_name)
  model_info_inputs = list(model_info.metadata.get("io_ports").get("inputs"))

  def _create_dummy_tensor(model_info_input):
    shape = model_info_input.get("shape")
    for i, _ in enumerate(shape):
      if isinstance(shape[i], str):
        shape[i] = 1
      key = jax.random.PRNGKey(0)
      return jax.random.normal(key, shape)

  inputs = [_create_dummy_tensor(item) for item in model_info_inputs]
  results = JaxBackend.run(model, inputs)
  logging.info("jax results shape is = %s", [x.shape for x in results])

  try:
    from onnxruntime import backend as OrtBackend  # pylint: disable=g-import-not-at-top

    ort_results = OrtBackend.run(model, np.asarray(inputs[0]))
    logging.info("ort results shape is = %s", [x.shape for x in results])
    similarity = jax.tree_util.tree_map(_cosin_sim, results, ort_results)
    print(f"jax and ort similarity = {similarity}")
  except ImportError as e:
    logging.warning(
        "No onnxruntime here. Skip the comparision, err message = %s", e
    )


def export_model():
  """export jax model to disk."""
  model_name_list = list(map(lambda x: x.model.lower(), hub.list_models()))
  model_name = FLAGS.model_name
  assert (
      model_name in model_name_list
  ), f"{model_name} is valid model name in onnx.hub."
  model = _download_model(model_name)
  model_info = hub.get_model_info(model_name)
  model_info_inputs = list(model_info.metadata.get("io_ports").get("inputs"))

  def _create_dummy_tensor(model_info_input):
    shape = model_info_input.get("shape")
    for i, _ in enumerate(shape):
      if isinstance(shape[i], str):
        shape[i] = 1
      key = jax.random.PRNGKey(0)
      return jax.random.normal(key, shape, dtype=jnp.float32)

  inputs = [_create_dummy_tensor(item) for item in model_info_inputs]
  input_names = [item["name"] for item in model_info_inputs]
  dict_inputs = {name: inp for name, inp in zip(input_names, inputs)}
  jax_model, params = call_onnx.call_onnx(model, inputs)
  output_names = [n.name for n in model.graph.output]

  # Wrap the model params and function into a JaxModule.
  jax_module = JaxModule(
      params,
      jax_model,
      trainable=False,
      # input_polymorphic_shape="(b, ...)" if batch_size is None else None,
  )

  # Specify the serving configuration and export the model.
  def tf_postprocessor(outputs):
    return {name: output for name, output in zip(output_names, outputs)}

  serving_configs = [
      ServingConfig(
          signature_key="serving_default",
          input_signature=[
              [
                  tf.TensorSpec(x.shape, x.dtype, name)
                  for x, name in zip(inputs, input_names)
              ]
          ],
          tf_postprocessor=tf_postprocessor,
      ),
  ]
  em = ExportManager(
      jax_module,
      serving_configs,
  )
  # Save the model.
  model_path = FLAGS.output_dir
  logging.info("Exporting the model to %s.", model_path)
  em.save(model_path)

  # Test that the saved model could be loaded and run.
  logging.info("Loading the model from %s.", model_path)
  loaded = tf.saved_model.load(model_path)

  savedmodel_dict_output = loaded.signatures["serving_default"](**dict_inputs)
  savedmodel_output = [savedmodel_dict_output[name] for name in output_names]
  jax_output = jax.jit(jax_model)(params, inputs)

  logging.info(
      "Savemodel output: %s, JAX output: %s",
      type(savedmodel_output),
      type(jax_output),
  )
  similarity = jax.tree_util.tree_map(_cosin_sim, jax_output, savedmodel_output)
  print(f"jax and savedmodel similarity = {similarity}")


def main(argv) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  logging.info("Start running model from onnx backend.")
  run_model_from_onnx_backend()
  logging.info("Start exporting model.")
  jax.config.update("jax2tf_default_native_serialization", True)
  jax.config.update("jax_enable_x64", True)
  export_model()


if __name__ == "__main__":
  app.run(main)
