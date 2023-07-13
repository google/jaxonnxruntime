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
"""global jaxonnxruntime configuration."""
import contextlib
import logging
import os
import threading
from typing import Any, Callable, List, Optional
# Borrow from jax/_src/config.py but removev all unnecceray flags support.
# pylint: disable=redefined-builtin,invalid-name,broad-exception-raised
# pylint: disable=missing-class-docstring,missing-function-docstring

logger = logging.getLogger(__name__)


def bool_env(varname: str, default: bool) -> bool:
  """Read an environment variable and interpret it as a boolean.

  True values are (case insensitive): 'y', 'yes', 't', 'true', 'on', and '1';
  false values are 'n', 'no', 'f', 'false', 'off', and '0'.

  Args:
    varname: the name of the variable
    default: the default boolean value

  Returns:
    True or False

  Raises: ValueError if the environment variable is anything else.
  """
  val = os.getenv(varname, str(default))
  val = val.lower()
  if val in ('y', 'yes', 't', 'true', 'on', '1'):
    return True
  elif val in ('n', 'no', 'f', 'false', 'off', '0'):
    return False
  else:
    raise ValueError(f'invalid truth value {val!r} for environment {varname!r}')


def int_env(varname: str, default: int) -> int:
  """Read an environment variable and interpret it as an integer."""
  return int(os.getenv(varname, str(default)))


class Config:
  _HAS_DYNAMIC_ATTRIBUTES = True

  def __init__(self):
    self.values = {}
    self.meta = {}
    self._update_hooks = {}

  def update(self, name, val):
    self.check_exists(name)
    if name not in self.values:
      raise Exception(f'Unrecognized config option: {name}')
    self.values[name] = val

    hook = self._update_hooks.get(name, None)
    if hook:
      hook(val)

  def read(self, name):
    return self._read(name)

  def _read(self, name):
    try:
      return self.values[name]
    except KeyError as e:
      raise AttributeError(f'Unrecognized config option: {name}') from e

  def add_option(
      self,
      name,
      default,
      opt_type,
      meta_args,
      meta_kwargs,
      update_hook: Optional[Callable[[Any], None]] = None,
  ):
    if name in self.values:
      raise Exception(f'Config option {name} already defined')
    self.values[name] = default
    self.meta[name] = (opt_type, meta_args, meta_kwargs)
    if update_hook:
      self._update_hooks[name] = update_hook
      update_hook(default)

  def check_exists(self, name):
    if name not in self.values:
      raise AttributeError(f'Unrecognized config option: {name}')

  def DEFINE_bool(self, name, default, *args, **kwargs):
    update_hook = kwargs.pop('update_hook', None)
    self.add_option(name, default, bool, args, kwargs, update_hook=update_hook)

  def DEFINE_integer(self, name, default, *args, **kwargs):
    update_hook = kwargs.pop('update_hook', None)
    self.add_option(name, default, int, args, kwargs, update_hook=update_hook)

  def DEFINE_float(self, name, default, *args, **kwargs):
    update_hook = kwargs.pop('update_hook', None)
    self.add_option(name, default, float, args, kwargs, update_hook=update_hook)

  def DEFINE_string(self, name, default, *args, **kwargs):
    update_hook = kwargs.pop('update_hook', None)
    self.add_option(name, default, str, args, kwargs, update_hook=update_hook)

  def DEFINE_enum(self, name, default, *args, **kwargs):
    update_hook = kwargs.pop('update_hook', None)
    self.add_option(
        name, default, 'enum', args, kwargs, update_hook=update_hook
    )

  def define_bool_state(
      self,
      name: str,
      default: bool,
      help: str,
      *,
      update_global_hook: Optional[Callable[[bool], None]] = None,
      update_thread_local_hook: Optional[
          Callable[[Optional[bool]], None]
      ] = None,
      extra_description: str = '',
  ):
    """Set up thread-local state and return a contextmanager for managing it."""
    name = name.lower()
    self.DEFINE_bool(
        name,
        bool_env(name.upper(), default),
        help,
        update_hook=update_global_hook,
    )

    def get_state(self):
      val = _thread_local_state.__dict__.get(name, unset)
      return val if val is not unset else self._read(name)

    setattr(Config, name, property(get_state))

    return _StateContextManager(
        name,
        help,
        update_thread_local_hook,
        extra_description=extra_description,
        default_value=True,
    )

  def define_enum_state(
      self,
      name: str,
      enum_values: List[str],
      default: Optional[str],
      help: str,
      update_global_hook: Optional[Callable[[str], None]] = None,
      update_thread_local_hook: Optional[
          Callable[[Optional[str]], None]
      ] = None,
  ):
    """Set up thread-local state and return a contextmanager for managing it."""
    name = name.lower()
    default = os.getenv(name.upper(), default)
    if default is not None and default not in enum_values:
      raise ValueError(f'Invalid value "{default}" for flag {name}')
    self.DEFINE_enum(
        name,
        default,
        enum_values=enum_values,
        help=help,
        update_hook=update_global_hook,
    )

    def get_state(self):
      val = _thread_local_state.__dict__.get(name, unset)
      return val if val is not unset else self._read(name)

    setattr(Config, name, property(get_state))

    def validate(new_val):
      if new_val is not None and (
          not isinstance(new_val, str) or new_val not in enum_values
      ):
        raise ValueError(
            f'new enum value must be None or in {enum_values}, '
            f'got {new_val} of type {type(new_val)}.'
        )

    return _StateContextManager(name, help, update_thread_local_hook, validate)

  def define_int_state(
      self,
      name: str,
      default: Optional[int],
      help: str,
      update_global_hook: Optional[Callable[[str], None]] = None,
      update_thread_local_hook: Optional[
          Callable[[Optional[str]], None]
      ] = None,
  ):
    """Set up thread-local state and return a contextmanager for managing it."""
    name = name.lower()
    default_env = os.getenv(name.upper(), default)
    if default_env is not None:
      try:
        default = int(default_env)
      except ValueError as e:
        raise ValueError(
            f'Invalid value "{default_env}" for flag {name}'
        ) from e
    self.DEFINE_integer(
        name, default, help=help, update_hook=update_global_hook
    )

    def get_state(self):
      val = _thread_local_state.__dict__.get(name, unset)
      return val if val is not unset else self._read(name)

    setattr(Config, name, property(get_state))

    def validate(new_val):
      if new_val is not None and not isinstance(new_val, int):
        raise ValueError(
            'new int config value must be None or of type int, '
            f'got {new_val} of type {type(new_val)}'
        )

    return _StateContextManager(name, help, update_thread_local_hook, validate)

  def define_float_state(
      self,
      name: str,
      default: Optional[float],
      help: str,
      update_global_hook: Optional[Callable[[str], None]] = None,
      update_thread_local_hook: Optional[
          Callable[[Optional[str]], None]
      ] = None,
  ):
    """Set up thread-local state and return a contextmanager for managing it."""
    name = name.lower()
    default_env = os.getenv(name.upper(), default)
    if default_env is not None:
      try:
        default = float(default_env)
      except ValueError as e:
        raise ValueError(
            f'Invalid value "{default_env}" for flag {name}'
        ) from e
    self.DEFINE_float(name, default, help=help, update_hook=update_global_hook)

    def get_state(self):
      val = _thread_local_state.__dict__.get(name, unset)
      return val if val is not unset else self._read(name)

    setattr(Config, name, property(get_state))

    def validate(new_val):
      if new_val is not None and not isinstance(new_val, (float, int)):
        raise ValueError(
            'new float config value must be None or of type float, '
            f'got {new_val} of type {type(new_val)}'
        )

    return _StateContextManager(name, help, update_thread_local_hook, validate)

  def define_string_state(
      self,
      name: str,
      default: Optional[str],
      help: str,
      update_global_hook: Optional[Callable[[str], None]] = None,
      update_thread_local_hook: Optional[
          Callable[[Optional[str]], None]
      ] = None,
  ):
    """Set up thread-local state and return a contextmanager for managing it."""

    def validate(new_val):
      if new_val is not None and not isinstance(new_val, str):
        raise ValueError(
            'new string config value must be None or of type str,'
            f' got {new_val} of type {type(new_val)}.'
        )

    return self.define_string_or_object_state(
        name,
        default,
        help,
        update_global_hook,
        update_thread_local_hook,
        validate,
    )

  def define_string_or_object_state(
      self,
      name: str,
      default: Any,
      help: str,
      update_global_hook: Optional[Callable[[Any], None]] = None,
      update_thread_local_hook: Optional[Callable[[Any], None]] = None,
      validate_new_val_hook: Optional[Callable[[Any], None]] = None,
  ):
    """Set up thread-local state and return a contextmanager for managing it."""
    name = name.lower()
    default = os.getenv(name.upper(), default)
    self.DEFINE_string(name, default, help=help, update_hook=update_global_hook)

    def get_state(self):
      val = _thread_local_state.__dict__.get(name, unset)
      return val if val is not unset else self._read(name)

    setattr(Config, name, property(get_state))

    return _StateContextManager(
        name, help, update_thread_local_hook, validate_new_val_hook
    )


class NoDefault:
  pass


no_default = NoDefault()


class _StateContextManager:

  def __init__(
      self,
      name,
      help,
      update_thread_local_hook,
      validate_new_val_hook: Optional[Callable[[Any], None]] = None,
      extra_description: str = '',
      default_value: Any = no_default,
  ):
    self._name = name
    self.__name__ = name
    self.__doc__ = (
        f'Context manager for `{name}` config option'
        f'{extra_description}.\n\n{help}'
    )
    self._update_thread_local_hook = update_thread_local_hook
    self._validate_new_val_hook = validate_new_val_hook
    self._default_value = default_value

  @contextlib.contextmanager
  def __call__(self, new_val: Any = no_default):
    if new_val is no_default:
      if self._default_value is not no_default:
        new_val = self._default_value  # default_value provided to constructor
      else:
        # no default_value provided to constructor and no value provided as an
        # argument, so we raise an error
        raise TypeError(
            f'Context manager for {self.__name__} config option '
            'requires an argument representing the new value for '
            'the config option.'
        )
    if self._validate_new_val_hook:
      self._validate_new_val_hook(new_val)
    prev_val = getattr(_thread_local_state, self._name, unset)
    setattr(_thread_local_state, self._name, new_val)
    if self._update_thread_local_hook:
      self._update_thread_local_hook(new_val)
    try:
      yield
    finally:
      if prev_val is unset:
        delattr(_thread_local_state, self._name)
        if self._update_thread_local_hook:
          self._update_thread_local_hook(None)
      else:
        setattr(_thread_local_state, self._name, prev_val)
        if self._update_thread_local_hook:
          self._update_thread_local_hook(prev_val)


_thread_local_state = threading.local()


class _Unset:
  pass


unset = _Unset()


class NameSpace:

  def __init__(self, getter, setter):
    # must use super because we override this class's __setattr__, see
    # https://docs.python.org/3/reference/datamodel.html#object.__setattr__
    super().__setattr__('_getter', getter)
    super().__setattr__('_setter', setter)

  def __getattr__(self, name):
    return self._getter(name)

  def __setattr__(self, name, val):
    self._setter(name, val)


config = Config()

jaxort_only_allow_initializers_as_static_args = config.define_bool_state(
    name='jaxort_only_allow_initializers_as_static_args',
    default=bool_env('JAXORT_ONLY_ALLOW_INITIALIZERS_AS_STATIC_ARGS', True),
    help=(
        'Some ONNX op inputs can not work under jax.jit. We need'
        'convert them to static arguments. jaxort_use_inputs_as_static_args'
        '=False make sure this static_args is from onnx initilizer but not'
        'depend on model inputs concrete value.'
    ),
)

jaxort_nonzero_use_fully_padding = config.define_bool_state(
    name='jaxort_nonzero_use_fully_padding',
    default=bool_env('JAXORT_NONZERO_USE_FULLY_PADDED', False),
    help=(
        'NonZero must provide static size attribute to support jax.jit.If this'
        ' option is True, we will use the input shape deduce the attributeif'
        ' users do not provide. This maybe wrong if it is not user'
        ' intention.The default value is False.Some models of zoo (like gpt-2)'
        ' need this walk-around.'
    ),
)

jaxort_if_op_reshape_output_for_llama = config.define_bool_state(
    name='jaxort_if_op_reshape_output_for_llama',
    default=False,
    help=(
        'The onnx If operator can have dynamic output shape, which cannot '
        'be supported by jax.jit. We have to manually manipulate the output '
        'shape if there is a mismatch between the outputs of else_branch and '
        'then_branch, as is the case for LLaMA.'
    ),
)

jaxort_experimental_support_abtract_input_shape = config.define_bool_state(
    name='jaxort_experimental_support_abtract_input_shape',
    default=False,
    help=(
        'Default behaviour is that call_onnx require real model input to'
        ' tracethe JAX function. If `jaxort_support_abtract_input_shape`, users'
        ' onlyneed provide input abstract shape and dtype info. Here we use'
        ' `jax.eval_shape`function to deduce the output shape and dtype'
    ),
)
