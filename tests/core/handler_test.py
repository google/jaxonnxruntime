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

from absl.testing import absltest
from jaxonnxruntime.core import handler
from onnx import defs


class TestHandler(absltest.TestCase):

  def test_get_since_version(self):
    class MyOpHandler(handler.Handler):
      pass

    MyOpHandler.DOMAIN = ""
    MyOpHandler.OP_TYPE = "Add"
    version = 11
    print("get_all_schemas", defs.get_all_schemas())
    schema = defs.get_schema(
        MyOpHandler.OP_TYPE,
        max_inclusive_version=version,
        domain=MyOpHandler.DOMAIN,
    )
    since_version = MyOpHandler.get_since_version(version)
    self.assertEqual(since_version, schema.since_version)

  def test_register_op(self):
    @handler.register_op("my_op", domain="ai.onnx")
    class MyOpHandler(handler.Handler):
      pass

    self.assertEqual(MyOpHandler.OP_TYPE, "my_op")
    self.assertEqual(MyOpHandler.DOMAIN, "ai.onnx")


if __name__ == "__main__":
  absltest.main()
