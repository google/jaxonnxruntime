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

"""Jax utility functions."""
from absl import logging
import jax


def eval_shape(fun, *args, **kwargs):
  """Evaluates a function and returns its outputs abstract shape."""
  if logging.vlog_is_on(3):
    logging.vlog(
        3, "jax_utils.eval_shape: args = %s, kwargs = %s", args, kwargs
    )
  return jax.eval_shape(fun, *args, **kwargs)
