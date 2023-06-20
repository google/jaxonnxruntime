# Contributing to jaxonnxruntime

🎉🎉 First off, thank you for taking the time to contribute! 🎉🎉

The following is a set of guidelines, but not rules, for contributing to jaxonnxruntime.
Use your best judgment, and feel free to propose changes to this document in a pull request.

We follow most of the best practices listed in the [contributing guidelines](https://github.com/google/flax/blob/main/docs/contributing.md) of the `google/flax` project .
Here we only list the difference.

<!-- ---

#### Table of Contents

- [Contributing to jaxonnxruntime](#contributing-to-jaxonnxruntime)
      - [Table of Contents](#table-of-contents)
  - [How to Contribute?](#how-to-contribute)
    - [Adding Support for New Operators](#adding-support-for-new-operators)

--- -->

## How to Contribute?

### Adding Support for New Operators

When running through a new onnx model, you may find that some operators are not implemented in our repository.
This can also be done by running the command:
```shell
$ python tools/analyze_model.py <onnx_model_path>
```
All source code for JAX backend of ONNX operators is located in [jaxonnxruntime/onnx_ops](https://github.com/google/jaxonnxruntime/tree/main/jaxonnxruntime/onnx_ops).
You can generate new template implementation of ONNX operators by running the following command, and then fill in the blanks marked by ```TODO```.
Please do use JAX methods to do the implementation!
```shell
$ python tools/op_code_generator.py <onnx_op_name>
```

After finishing the implementation, please add unit test for the new operator by adding a line
```python
include_patterns.append('test_{op_name_lower}_')
```
to [```tests/onnx_ops_test.py```](https://github.com/google/jaxonnxruntime/blob/main/tests/onnx_ops_test.py).
Then run the unit tests with the following command to make sure it is compatible with both ONNX and JAX.
Make sure all tests are passed before submitting a pull request.
```shell
$ python tests/onnx_ops_test.py
```
