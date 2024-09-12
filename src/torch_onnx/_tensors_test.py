# mypy: allow-untyped-defs
from __future__ import annotations

import unittest

import onnxscript

from torch_onnx import _tensors


class SymbolicTensorTest(unittest.TestCase):
    def test_it_is_hashable(self):
        tensor = _tensors.SymbolicTensor(opset=onnxscript.values.Opset(domain="test", version=1))
        self.assertEqual(hash(tensor), hash(tensor))
        self.assertIn(tensor, set([tensor]))


if __name__ == "__main__":
    unittest.main()
