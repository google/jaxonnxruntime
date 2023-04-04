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

"""Create the Backend class."""
# pylint: disable=unused-argument

from typing import Any, NewType

import jax
from jaxonnxruntime import call_onnx

import onnx


# Copy from onnx.backend base.py.
# Due to some reasons, we can not inherit base.py.


class DeviceType:
  """An enumeration of device types."""

  _TYPE = NewType('_TYPE', int)
  CPU: _TYPE = _TYPE(0)
  CUDA: _TYPE = _TYPE(1)
  TPU: _TYPE = _TYPE(2)


class BackendRep:
  """Provides an interface for executing an ONNX model on the JAX backend."""

  def __init__(self, model: onnx.ModelProto) -> None:
    """Initializes a new instance of the ONNX backend representation class.

    Args:
      model: The ONNX model to represent.
    """
    self._model = model

  def run(self, inputs: Any, **kwargs: Any) -> tuple[Any, ...]:
    """Runs the ONNX model using the provided input data.

    Args:
      inputs: The input data for the model.
      **kwargs: Additional keyword arguments to pass to the underlying
        `call_onnx` function.

    Returns:
      A tuple (Any, ...) of output data produced by the model.
    """
    model_func, model_params = call_onnx.call_onnx(
        self._model, inputs, **kwargs
    )
    return model_func(model_params=model_params, inputs=inputs)


class Backend:
  """Backend is the entity that will take an ONNX model with inputs.

  perform a computation, and then return the output.
  For one-off execution, users can use run_node and run_model to obtain results
  quickly.
  For repeated execution, users should use prepare, in which the Backend
  does all of the preparation work for executing the model repeatedly
  (e.g., loading initializers), and returns a BackendRep handle.
  """

  @classmethod
  def is_compatible(cls, model: onnx.ModelProto, device: str = 'CPU') -> bool:
    """Returns whether the ONNX model is compatible.

    Args:
      model: The ONNX model to check for compatibility.
      device: The name of the backend device to check for compatibility.
        Defaults to 'CPU'.

    Returns:
      True if the model is compatible with the backend device, False otherwise.
    """
    # Check if the specified device is available
    if device == 'CUDA':
      try:
        jax.devices('gpu')
      except:  # pylint: disable=bare-except
        return False
    if device == 'TPU':
      try:
        jax.devices('tpu')
      except:  # pylint: disable=bare-except
        return False
    return True

  @classmethod
  def prepare(
      cls, model: onnx.ModelProto, device: str = 'CPU', **kwargs: Any
  ) -> BackendRep:
    """Prepares an ONNX model for execution on the specified backend device.

    Args:
      model: The ONNX model to prepare.
      device: The name of the backend device to use for execution. Defaults to
        'CPU'.
      **kwargs: Additional keyword arguments to pass to the `BackendRep`
        constructor.

    Returns:
      An instance of the `BackendRep` class representing the
      prepared model.
    """
    return BackendRep(model)

  @classmethod
  def run_model(
      cls,
      model: onnx.ModelProto,
      inputs: Any,
      device: str = 'CPU',
      **kwargs: Any,
  ) -> tuple[Any, ...]:
    """Runs the provided ONNX model using the specified backend device.

    Args:
        model: The ONNX model to run.
        inputs: The input data for the model.
        device: The name of the backend device to use for execution. Defaults to
          'CPU'.
        **kwargs: Additional keyword arguments to pass to the `prepare` method.

    Returns:
        A tuple (Any, ...) of output data produced by the model.

    Raises:
        RuntimeError: If the backend fails to prepare the model for execution.
    """
    backend = cls.prepare(model, device, **kwargs)
    if not backend:
      raise RuntimeError(
          'Failed to prepare the model for execution on the backend.'
      )
    return backend.run(inputs)

  @classmethod
  def supports_device(cls, device: str) -> bool:
    """Checks whether the backend is compiled with support for the specified device.

    Args:
      device: The name of the device to check support for.

    Returns:
      True if the backend supports the specified device, False otherwise.
    """
    return True


run_model = Backend.run_model
run = Backend.run_model
