# JAX ONNX Runtime

JAX ONNX Runtime is a robust and user-friendly tool chain that enables the seamless execution of ONNX models using JAX as the backend.

More specifically, this tool chain has the abilities:

- ONNX Model Conversion: Converts ONNX models into JAX format modules. Tested on popular large language models including GPT-2, BERT, and LLaMA.

- Hardware Acceleration: Enable the jit mode of the converted JAX modules, which accelerates execution on GPU and/or TPU.

- Compatibility with JAX ecosystem: E.g., export models by Orbax, and serve the saved models by Tensorflow Serving system.

## Get Started

- We follow most of the interface definitions by `onnx.backend` [here](https://onnx.ai/onnx/api/backend.html).

- Please check a brief example on model conversion and forward calling in [`examples/imagenet/imagenet_main.py`](https://github.com/google/jaxonnxruntime/blob/main/examples/imagenet/imagenet_main.py).

## Contributions and Discussions

Thank you for taking the time to contribute! Please see [the contribution guidelines](https://github.com/google/jaxonnxruntime/blob/main/contributing.md).

## License

This project is licensed under the [Apache License](https://github.com/google/jaxonnxruntime/blob/main/LICENSE).
