# Copyright 2024 The Jaxonnxruntime Authors.
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

"""Test orbax.checkpoint."""

import os
from absl.testing import absltest
import chex
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp


class OrbaxCheckpointTest(absltest.TestCase):

  def test_basic(self):
    tmp_dir = os.getenv(
        'TEST_UNDECLARED_OUTPUTS_DIR', self.create_tempdir().full_path
    )
    my_tree = {
        'a': np.arange(8),
        'b': {
            'c': 42,
            'd': np.arange(16),
        },
    }
    my_tree = jax.tree.map(jnp.array, my_tree)
    checkpointer = ocp.StandardCheckpointer()

    def to_shape_dtype_struct(x):
      shape = x.shape
      dtype = x.dtype
      sharding = x.sharding if hasattr(x, 'sharding') else None
      return jax.ShapeDtypeStruct(shape=shape, dtype=dtype, sharding=sharding)

    abstract_my_tree = jax.tree.map(to_shape_dtype_struct, my_tree)

    with self.subTest('test_1'):
      ckpt_path = os.path.join(tmp_dir, 'test_1')
      checkpointer.save(ckpt_path, my_tree)
      new_tree = checkpointer.restore(ckpt_path, abstract_my_tree)
      chex.assert_trees_all_equal(my_tree, new_tree)

    with self.subTest('test_2'):
      my_tree_flatten = jax.tree.leaves(my_tree)
      ckpt_path = os.path.join(tmp_dir, 'test_2')
      checkpointer.save(ckpt_path, my_tree_flatten)
      abstract_my_tree_flatten = jax.tree.leaves(abstract_my_tree)
      new_tree_flatten = checkpointer.restore(
          ckpt_path, abstract_my_tree_flatten
      )
      my_tree_flatten = jax.tree.leaves(my_tree)
      chex.assert_trees_all_equal(my_tree_flatten, new_tree_flatten)


if __name__ == '__main__':
  absltest.main()
