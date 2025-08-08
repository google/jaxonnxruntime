# Copyright 2025 The Jaxonnxruntime Authors.
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

"""JAX based onnxruntime."""

# pylint: disable=g-importing-member
from jaxonnxruntime import backend
from jaxonnxruntime import version
from jaxonnxruntime.core import call_onnx
from jaxonnxruntime.core import config_class

# pylint: disable=redefined-builtin
# pylint: disable=g-bad-import-order
from jaxonnxruntime.onnx_ops import abs
from jaxonnxruntime.onnx_ops import acos
from jaxonnxruntime.onnx_ops import acosh
from jaxonnxruntime.onnx_ops import add
from jaxonnxruntime.onnx_ops import and_op
from jaxonnxruntime.onnx_ops import argmax
from jaxonnxruntime.onnx_ops import argmin
from jaxonnxruntime.onnx_ops import asin
from jaxonnxruntime.onnx_ops import asinh
from jaxonnxruntime.onnx_ops import atan
from jaxonnxruntime.onnx_ops import atanh
from jaxonnxruntime.onnx_ops import averagepool
from jaxonnxruntime.onnx_ops import batchnormalization
from jaxonnxruntime.onnx_ops import bitshift
from jaxonnxruntime.onnx_ops import cast
from jaxonnxruntime.onnx_ops import castlike
from jaxonnxruntime.onnx_ops import ceil
from jaxonnxruntime.onnx_ops import clip
from jaxonnxruntime.onnx_ops import concat
from jaxonnxruntime.onnx_ops import constant
from jaxonnxruntime.onnx_ops import constantofshape
from jaxonnxruntime.onnx_ops import conv
from jaxonnxruntime.onnx_ops import cos
from jaxonnxruntime.onnx_ops import cosh
from jaxonnxruntime.onnx_ops import dequantizelinear
from jaxonnxruntime.onnx_ops import div
from jaxonnxruntime.onnx_ops import dropout
from jaxonnxruntime.onnx_ops import einsum
from jaxonnxruntime.onnx_ops import elu
from jaxonnxruntime.onnx_ops import equal
from jaxonnxruntime.onnx_ops import erf
from jaxonnxruntime.onnx_ops import exp
from jaxonnxruntime.onnx_ops import expand
from jaxonnxruntime.onnx_ops import flatten
from jaxonnxruntime.onnx_ops import gather
from jaxonnxruntime.onnx_ops import gatherelements
from jaxonnxruntime.onnx_ops import gemm
from jaxonnxruntime.onnx_ops import globalaveragepool
from jaxonnxruntime.onnx_ops import greater
from jaxonnxruntime.onnx_ops import greaterorequal
from jaxonnxruntime.onnx_ops import hardsigmoid
from jaxonnxruntime.onnx_ops import identity
from jaxonnxruntime.onnx_ops import if_op
from jaxonnxruntime.onnx_ops import leakyrelu
from jaxonnxruntime.onnx_ops import less
from jaxonnxruntime.onnx_ops import lessorequal
from jaxonnxruntime.onnx_ops import log
from jaxonnxruntime.onnx_ops import logsoftmax
from jaxonnxruntime.onnx_ops import lrn
from jaxonnxruntime.onnx_ops import matmul
from jaxonnxruntime.onnx_ops import max
from jaxonnxruntime.onnx_ops import maxpool
from jaxonnxruntime.onnx_ops import min
from jaxonnxruntime.onnx_ops import mul
from jaxonnxruntime.onnx_ops import neg
from jaxonnxruntime.onnx_ops import nonzero
from jaxonnxruntime.onnx_ops import onehot
from jaxonnxruntime.onnx_ops import onnx_not
from jaxonnxruntime.onnx_ops import or_op
from jaxonnxruntime.onnx_ops import pad
from jaxonnxruntime.onnx_ops import pow
from jaxonnxruntime.onnx_ops import prelu
from jaxonnxruntime.onnx_ops import quantizelinear
from jaxonnxruntime.onnx_ops import range
from jaxonnxruntime.onnx_ops import reciprocal
from jaxonnxruntime.onnx_ops import reducemax
from jaxonnxruntime.onnx_ops import reducemean
from jaxonnxruntime.onnx_ops import reducesum
from jaxonnxruntime.onnx_ops import relu
from jaxonnxruntime.onnx_ops import reshape
from jaxonnxruntime.onnx_ops import scatterelements
from jaxonnxruntime.onnx_ops import scatternd
from jaxonnxruntime.onnx_ops import selu
from jaxonnxruntime.onnx_ops import shape
from jaxonnxruntime.onnx_ops import shrink
from jaxonnxruntime.onnx_ops import sigmoid
from jaxonnxruntime.onnx_ops import sign
from jaxonnxruntime.onnx_ops import sin
from jaxonnxruntime.onnx_ops import sinh
from jaxonnxruntime.onnx_ops import slice
from jaxonnxruntime.onnx_ops import softmax
from jaxonnxruntime.onnx_ops import softplus
from jaxonnxruntime.onnx_ops import split
from jaxonnxruntime.onnx_ops import sqrt
from jaxonnxruntime.onnx_ops import squeeze
from jaxonnxruntime.onnx_ops import sub
from jaxonnxruntime.onnx_ops import sum
from jaxonnxruntime.onnx_ops import tan
from jaxonnxruntime.onnx_ops import tanh
from jaxonnxruntime.onnx_ops import tile
from jaxonnxruntime.onnx_ops import topk
from jaxonnxruntime.onnx_ops import transpose
from jaxonnxruntime.onnx_ops import trilu
from jaxonnxruntime.onnx_ops import unsqueeze
from jaxonnxruntime.onnx_ops import where
# pylint: enable=g-bad-import-order

Backend = backend.Backend
config = config_class.config
__version__ = version.__version__
