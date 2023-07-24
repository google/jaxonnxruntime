Adding a New Op
=======================================

All JAX backend implementations for ONNX operators inherit from a base class Handler.
The purpose is to provide a way to register new op implementations for different versions of ONNX.
Users can use the following template to implement ops.

It is required to finish the following three steps to add a new operator backend:

1. Implement a JAX backend for the operator

2. Add imports to ``jaxonnxruntime/onnx_ops/__init__.py``

3. Add corresponding unit tests to ``tests/onnx_ops_test.py``

A Shortcut to Implementing Ops
---------------------------------------

The library provides operator implementation templates in `tools/op_code_generator.py`.
Just run the command below in terminal, and we can get the templates in the workspace::

    $ python tools/op_code_generator.py OP_NAME

It will generate a template to implement the JAX backend, and also automatically
add imports to ``jaxonnxruntime/onnx_ops/__init__.py``.
For unit test cases, it only generates recommended test names to be manually added to ``tests/onnx_ops_test.py``.

Base Class Handler
----------------------------------------

The Handler class is the base class for all operator implementations.
It contains the following class attributes, which are used as identifiers for individual ops:

* ``DOMAIN``: the domain of the op
* ``OP_TYPE``: the type of the op
* ``SINCE_VERSION``: the version since which the op is available

It also contains the following class methods:

* ``get_since_version(cls, version)``: a method to get ``SINCE_VERSION`` based on the version of the ONNX opset being used. Returns an integer.

Example::

    class Handler:
      DOMAIN = ""
      OP_TYPE = ""
      SINCE_VERSION = 0

    def get_since_version(cls, version) -> int:
      # return since_version

Here ``version`` is the model opset version specified by users
(see `here <https://github.com/onnx/onnx/blob/main/docs/Versioning.md#operator-sets>`_).
The result ``since_version`` is deduced from the opset version.

We use python`s ``__subclass__`` to collect all defined ONNX Op by the following code::

  for handler in Handler.__subclasses__():
    domain = hanlder.DOMAIN
    op_type = handler.OP_TYPE
    handler.VERSION = model.opset.version
    since_version = get_since_version()
    handler.SINCE_VERSION = since_version
    impl_func = get_attr(handler, f"version_{since_version}")

In subclasses where operators are implemented, it is needed to define different individual
call function ``version_{since_version}`` for different ``since_version``.

Op Implementation in Subclasses
------------------------------------------

Preprocessing in ``_prepare``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before returning the actual JAX implementation of the operator, we use ``_prepare`` to do necessary preprocessing.
The main purpose is to rewrite ``OnnxNode.attrs_dict``, which will be used as kwargs to the actual JAX implementation.
The items in ``OnnxNode.attrs_dict`` generally come from the original ``onnx.NodeProto.attributes``,
but in some special cases, we also save information to this dictionary from dummy inputs during tracing.
We take advantage of this preprocessing to set default values of those node attributes,
or manipulate dtype to make it compatible with JAX implementations wrapped in ``jax.jit``.

Take `version 13 Softmax`_ as an example. It has one attribute: ``axis`` with default is -1.
The attribute is added to ``OnnxNode.attrs_dict`` in its ``_prepare`` as below::

  @handler.register_op('Softmax')
  class Softmax(handler.Handler):
    """Implementation of the ONNX Softmax operator."""

    @classmethod
    def _prepare(
        cls,
        node: onnx_node.OnnxNode,
        inputs: Sequence[Any],
        onnx_jax_impl: Any,
    ):
      node.attrs_dict['axis'] = node.attrs.get('axis', -1)

JAX Implementations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finally we get to the actual implementation of the operator by JAX.
In our codebase, all JAX implementations of ONNX ops are named as ``onnx_{op_name}``.
We expect all JAX implementations can be wrapped in ``jax.jit`` to improve performance.
Usually node attributes are set to be static arguments to the jit functions.
But for some special ops such as ``Reshape`` and ``NonZero``,
the shape of outputs depends on some inputs which is not allowed by ``jax.jit``.
Therefore, we may also use part of the inputs as static arguments for them.

Take `version 13 Softmax`_ for an example again. Its JAX implementation is listed below::

  @functools.partial(jit, static_argnames=('axis',))
  def onnx_softmax(*input_args, axis):
    assert len(input_args) == 1
    x = input_args[0]
    return jax.nn.softmax(x, axis=axis)

Return the JAX Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The JAX backend function is returned by ``version_{since_version}``. For example::

  @handler.register_op('Softmax')
  class Softmax(handler.Handler):
    """Implementation of the ONNX Softmax operator."""
    ...

    @classmethod
    def version_13(
        cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
    ) -> Callable[..., Any]:
      """ONNX version_13 Softmax op."""
      cls._prepare(node, inputs, onnx_softmax)
      return onnx_softmax

If there are multiple versions of the op,
we should return them in individual ``version_{since_version}`` functions.

Register New Op in ``__init__``
-------------------------------------------------

Please don`t forget to add the following line to onnx_ops/__init__.py.
This can also be automatically done by ``tools/op_code_generator.py``.

.. code-block::

    from . import OP_NAME_LOWER as OP_NAME_LOWER

Add Unit Test for the Implementation
-------------------------------------------------

We also wanted to include all operators in the test suite.
Please add the following line to ``tests/onnx_ops_test.py``.
Make sure the prefix is correct and aligns with the test cases provided by onnx backend test suite.

.. code-block::

    include_patterns.append('test_{OP_NAME}_')

.. _version 13 Softmax: https://github.com/onnx/onnx/blob/main/docs/Operators.md#Softmax
