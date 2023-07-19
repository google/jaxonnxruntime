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

"""ONNX node test."""

import collections
from typing import Any

from absl.testing import absltest
import jax
from jaxonnxruntime import config
from jaxonnxruntime import runner
from jaxonnxruntime.backend import Backend as JaxBackend  # pylint: disable=g-importing-member


jax.config.update('jax_enable_x64', True)
jax.config.update('jax_numpy_rank_promotion', 'warn')
config.update('jaxort_only_allow_initializers_as_static_args', False)


class Runner(runner.Runner):

  def __init__(
      self, backend: type(JaxBackend), parent_module: Any = None
  ) -> None:
    self.backend = backend
    self._parent_module = parent_module
    self._include_patterns = set()  # type: ignore[var-annotated]
    self._exclude_patterns = set()  # type: ignore[var-annotated]
    self._xfail_patterns = set()  # type: ignore[var-annotated]
    self._test_items = collections.defaultdict(dict)  # type: ignore[var-annotated]

    for rt in runner.load_model_tests(kind='node'):
      self._add_model_test(rt, 'Node')

    for rt in runner.load_model_tests(kind='simple'):
      self._add_model_test(rt, 'Simple')

    for ct in runner.load_model_tests(kind='pytorch-converted'):
      self._add_model_test(ct, 'PyTorchConverted')

    for ot in runner.load_model_tests(kind='pytorch-operator'):
      self._add_model_test(ot, 'PyTorchOperator')


class NodeTest(absltest.TestCase):
  pass


backend_test = Runner(JaxBackend, __name__)
expect_fail_patterns = []
include_patterns = []
exclude_patterns = []

include_patterns.append('test_abs_')
include_patterns.append('test_acos_')
include_patterns.append('test_acosh_')
include_patterns.append('test_add_')
include_patterns.append('test_and_')
include_patterns.append('test_argmax_')
include_patterns.append('test_argmin_')
include_patterns.append('test_asin_')
include_patterns.append('test_asinh_')
include_patterns.append('test_atan_')
include_patterns.append('test_atanh_')
include_patterns.append('test_averagepool_')
include_patterns.append('test_batchnormalization_')
include_patterns.append('test_bitshift_')
include_patterns.append('test_cast_')
include_patterns.append('test_castlike_')
include_patterns.append('test_ceil_')
include_patterns.append('test_clip_')
include_patterns.append('test_concat_')
include_patterns.append('test_constant_')
include_patterns.append('test_constantofshape_')
include_patterns.append('test_conv_')
include_patterns.append('test_cos_')
include_patterns.append('test_cosh_')
include_patterns.append('test_div_')
include_patterns.append('test_dropout_')
include_patterns.append('test_einsum_')
include_patterns.append('test_equal_')
include_patterns.append('test_erf_')
include_patterns.append('test_exp_')
include_patterns.append('test_expand_')
include_patterns.append('test_flatten_')
include_patterns.append('test_gather_')
include_patterns.append('test_gather_elements_')
include_patterns.append('test_gemm_')
include_patterns.append('test_globalaveragepool_')
include_patterns.append('test_identity_')
include_patterns.append('test_if_')
include_patterns.append('test_leakyrelu_')
include_patterns.append('test_less_')
include_patterns.append('test_lessorequal_')
include_patterns.append('test_log_')
include_patterns.append('test_logsoftmax_')
include_patterns.append('test_lrn_')
include_patterns.append('test_matmul_')
include_patterns.append('test_max_')
include_patterns.append('test_maxpool_')
include_patterns.append('test_min_')
include_patterns.append('test_mul_')
include_patterns.append('test_neg_')
include_patterns.append('test_onehot_')
include_patterns.append('test_or_')
include_patterns.append('test_pow_')
include_patterns.append('test_prelu_')
include_patterns.append('test_range_')
include_patterns.append('test_reciprocal_')
include_patterns.append('test_reduce_max_')
include_patterns.append('test_reduce_mean_')
include_patterns.append('test_reduce_sum_')
include_patterns.append('test_relu_')
include_patterns.append('test_reshape_')
include_patterns.append('test_selu_')
include_patterns.append('test_shape_')
include_patterns.append('test_sigmoid_')
include_patterns.append('test_sin_')
include_patterns.append('test_sinh_')
include_patterns.append('test_slice_')
include_patterns.append('test_softmax_')
include_patterns.append('test_softplus_')
include_patterns.append('test_split_')
include_patterns.append('test_sqrt_')
include_patterns.append('test_sub_')
include_patterns.append('test_sum_')
include_patterns.append('test_squeeze_')
include_patterns.append('test_tanh_')
include_patterns.append('test_top_k_')
include_patterns.append('test_transpose_')
include_patterns.append('test_tri')
include_patterns.append('test_unsqueeze_')
include_patterns.append('test_where_')


# TODO(johnqiangzhang): should modify onnx.numpy_helper.to_array to support load
# bfloat16.
exclude_patterns.append('test_cast_FLOAT_to_BFLOAT16')
exclude_patterns.append('test_castlike_FLOAT_to_BFLOAT16')
# Not implement yet
exclude_patterns.append('test_gather_elements_')  # Op GatherElements
exclude_patterns.append('test_reduce_sum_square_')  # Op ReduceSumSquare
exclude_patterns.append('test_leakyrelu_default_expanded_cpu')  # Op CastLike
exclude_patterns.append('test_leakyrelu_example_expanded_cpu')  # Op CastLlike
exclude_patterns.append('test_leakyrelu_expanded_cpu')  # Op CastLike
exclude_patterns.append('test_relu_expanded_ver18_cpu')  # Op CastLike
exclude_patterns.append('test_split_to_sequence_1_cpu')  # Op SplitToSequence
exclude_patterns.append('test_split_to_sequence_2_cpu')  # Op SplitToSequence
exclude_patterns.append(
    'test_split_to_sequence_nokeepdims_cpu'
)  # Op SplitToSequence
# Clip: In ONNX's def, the Clip op's input `min` and `max` are optional.
# The input name list would be ['data', 'min', 'max'].
# In the test cases below, when `min` is not given, its name is set to
# be an empty string. But this is not compatible with jaxonnxruntime,
# as we don't handle placeholder as input names if the actual input data is
# missing. There will be a mismatch between the length of input name list and
# input data list. Therefore, we only include test cases with all three inputs,
# and with the first two inputs. Test cases with input name list
# ['data', '', 'max'] are excluded.
exclude_patterns.append('test_clip_default_inbounds_')
exclude_patterns.append('test_clip_default_int8_inbounds_')
exclude_patterns.append('test_clip_default_int8_max_')
exclude_patterns.append('test_clip_default_max_')
# Need more debug
exclude_patterns.append('test_softmax_axis_0_expanded_cpu')
exclude_patterns.append('test_softmax_axis_1_expanded_cpu')
exclude_patterns.append('test_softmax_axis_2_expanded_cpu')
exclude_patterns.append('test_softmax_default_axis_expanded_cpu')
exclude_patterns.append('test_softmax_large_number_expanded_cpu')
exclude_patterns.append('test_softmax_negative_axis_expanded_cpu')
exclude_patterns.append('test_range_float_type_positive_delta_expanded_cpu')
exclude_patterns.append('test_range_float_type_positive_delta_expanded_gpu')
exclude_patterns.append('test_range_int32_type_negative_delta_expanded_cpu')
exclude_patterns.append('test_range_int32_type_negative_delta_expanded_gpu')
# The following four requires `SequenceConstruct` Op. Will add them in the
# future.
exclude_patterns.append('test_if_opt_cpu')
exclude_patterns.append('test_if_seq_cpu')
exclude_patterns.append('test_if_opt_gpu')
exclude_patterns.append('test_if_seq_gpu')
# Not implement yet
exclude_patterns.append('test_maxpool_with_argmax_2d_')

expect_fail_patterns.extend([
    # cast
    'test_cast_FLOAT_to_STRING',
    'test_cast_STRING_to_FLOAT',
    'test_cast_FLOAT16_to_FLOAT8E4M3FNUZ_',
    'test_cast_FLOAT16_to_FLOAT8E4M3FNUZ_cpu',
    'test_cast_FLOAT16_to_FLOAT8E4M3FN_cpu',
    'test_cast_FLOAT16_to_FLOAT8E5M2FNUZ_cpu',
    'test_cast_FLOAT16_to_FLOAT8E5M2_cpu',
    'test_cast_FLOAT_to_FLOAT8E4M3FNUZ_cpu',
    'test_cast_FLOAT_to_FLOAT8E4M3FN_cpu',
    'test_cast_FLOAT_to_FLOAT8E5M2FNUZ_cpu',
    'test_cast_FLOAT_to_FLOAT8E5M2_cpu',
    'test_cast_no_saturate_FLOAT16_to_FLOAT8E4M3FNUZ_cpu',
    'test_cast_no_saturate_FLOAT16_to_FLOAT8E4M3FN_cpu',
    'test_cast_no_saturate_FLOAT16_to_FLOAT8E5M2FNUZ_cpu',
    'test_cast_no_saturate_FLOAT16_to_FLOAT8E5M2_cpu',
    'test_cast_no_saturate_FLOAT_to_FLOAT8E4M3FNUZ_cpu',
    'test_cast_no_saturate_FLOAT_to_FLOAT8E4M3FN_cpu',
    'test_cast_no_saturate_FLOAT_to_FLOAT8E5M2FNUZ_cpu',
    'test_cast_no_saturate_FLOAT_to_FLOAT8E5M2_cpu',
    # castlike
    'test_castlike_FLOAT_to_STRING',
    'test_castlike_STRING_to_FLOAT',
    'test_castlike_FLOAT16_to_FLOAT8E4M3FNUZ_',
    'test_castlike_FLOAT16_to_FLOAT8E4M3FNUZ_',
    'test_castlike_FLOAT16_to_FLOAT8E4M3FN_',
    'test_castlike_FLOAT16_to_FLOAT8E5M2FNUZ_',
    'test_castlike_FLOAT16_to_FLOAT8E5M2_',
    'test_castlike_FLOAT_to_FLOAT8E4M3FNUZ_',
    'test_castlike_FLOAT_to_FLOAT8E4M3FN_',
    'test_castlike_FLOAT_to_FLOAT8E5M2FNUZ_',
    'test_castlike_FLOAT_to_FLOAT8E5M2_',
    'test_castlike_no_saturate_FLOAT16_to_FLOAT8E4M3FNUZ_',
    'test_castlike_no_saturate_FLOAT16_to_FLOAT8E4M3FN_',
    'test_castlike_no_saturate_FLOAT16_to_FLOAT8E5M2FNUZ_',
    'test_castlike_no_saturate_FLOAT16_to_FLOAT8E5M2_',
    'test_castlike_no_saturate_FLOAT_to_FLOAT8E4M3FNUZ_',
    'test_castlike_no_saturate_FLOAT_to_FLOAT8E4M3FN_',
    'test_castlike_no_saturate_FLOAT_to_FLOAT8E5M2FNUZ_',
    'test_castlike_no_saturate_FLOAT_to_FLOAT8E5M2_',
    'test_castlike_.*_expanded_',
    # others
    'test_maxpool_2d_ceil_',
    'test_averagepool_2d_ceil_',
    'test_averagepool_2d_dilations_',
    'test_nonzero_',
    # np.object is not valid type for jax.array
    'test_equal_string_',
])


for pattern in include_patterns:
  backend_test.include(pattern)

for pattern in exclude_patterns:
  backend_test.exclude(pattern)

for pattern in expect_fail_patterns:
  backend_test.xfail(pattern)

# import all test cases at global scope to make them visible to python.unittest
globals().update(backend_test.test_cases)


if __name__ == '__main__':
  absltest.main()
