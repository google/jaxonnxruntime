# Contributing to jaxonnxruntime

🎉🎉 First off, thank you for taking the time to contribute! 🎉🎉

The following is a set of guidelines, but not rules, for contributing to jaxonnxruntime.
Use your best judgment, and feel free to propose changes to this document in a pull request.

---

#### Table of Contents

[Getting Started](#getting-started)

[What Kind of Contributions Are Welcome?](#what-kind-of-contributions-are-welcome)

[How to Contribute?](#how-to-contribute)
* [Reporting Bugs](#reporting-bugs)
* [Adding Support for New Operators](#adding-support-for-new-operators)
* [Code Review](#code-review)

[Code of Conduct](#code-of-conduct)

---

## Getting Started

To get started contributing, please:

1. Fork the repository to your own GitHub account.
2. Clone the repository to your local machine.
3. Create a branch locally with a succinct but descriptive name.
4. Commit changes to the branch.
5. Follow any formatting and testing guidelines specific to this repo.
6. Push changes to your fork.
7. Open a pull request in our repository and follow the pull request template so that we can efficiently review the changes.

## What Kind of Contributions Are Welcome?

We welcome all kinds of contributions, including:

* Bug fixes
* New features
* Documentation improvements
* Translations

## How to Contribute?

### Reporting Bugs

<!--- Maybe add a bug report template? --->
Bugs are tracked as [Github Issues](https://docs.github.com/en/issues/tracking-your-work-with-issues/about-issues).
Please explain the problem and include additional details to help maintainers reproduce the problem:

* Use a clear and descriptive title.
* Describe the exact steps which reproduce the problem in as many details as possible.
* Provide specific examples to demonstrate the steps.
* Describe the behavior you observed after following the steps and point out what exactly is the problem with that behavior.
* Explain which behavior you expected to see instead and why.

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

### Code Review

All submissions, including submissions by project members, require review. We use GitHub pull requests for this purpose.
Consult [GitHub Help](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) for more information on using pull requests.
<!--- Maybe add a pull request template? --->

## Code of Conduct

This project follows [Google's Open Source Community Guidelines](https://opensource.google/conduct/).
Please follow the guidelines in all your interactions with the project community.

Thank you for contributing!
