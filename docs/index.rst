.. Jaxonnxruntime documentation main file, created by
   sphinx-quickstart on Mon Feb 17 11:41:38 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

******************************
Jaxonnxruntime
******************************


.. div:: sd-text-left sd-font-italic

   JAX based ONNX backend


----

`Jaxonnxruntime` is focused on creating a JAX-based backend for the ONNX format. The benefits of using ONNX include interoperability and ease of hardware access, while JAX provides a similar API to Numpy and allows for performance speed-ups through jit compilation.

`Jaxonnxruntime` implements the backend by re-writing the ONNX operator implementations in the "JAX programming way" and interpreting all data structures as PyTree. The user will be able to run the jit function on the run_model function for performance speed-up and apply other Jax transformations.
