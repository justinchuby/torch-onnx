# ruff: noqa: UP037

import unittest

from torch_onnx import _core
import torch





bf16 = torch.bfloat16
f64 = torch.float64
f32 = torch.float32
f16 = torch.float16
c32 = torch.complex32
c64 = torch.complex64
c128 = torch.complex128
i8 = torch.int8
i16 = torch.int16
i32 = torch.int32
i64 = torch.int64
b8 = torch.bool
u8 = torch.uint8
u16 = torch.uint16
u32 = torch.uint32
u64 = torch.uint64



class ExportedProgramToIrTest(unittest.TestCase):
    def test_output_metadata_with_tuple_outputs(self):
        class GraphModule(torch.nn.Module):
            def forward(self, arg0_1: "f32[4, 3]", arg1_1: "f32[4, 3]", arg2_1: "i64[4]"):
                # File: /Users/justinc/Documents/GitHub/torch-onnx/tests/torch_tests/torch_onnx_test.py:9519 in forward, code: return self.emb(input), self.emb2(input)
                embedding: "f32[4, 3]" = torch.ops.aten.embedding.default(arg0_1, arg2_1, 1);  arg0_1 = None
                embedding_1: "f32[4, 3]" = torch.ops.aten.embedding.default(arg1_1, arg2_1, 1);  arg1_1 = arg2_1 = None
                return (embedding, embedding_1)

        exported_program = torch.export.export(GraphModule(), (torch.rand(4, 3), torch.rand(4, 3), torch.rand(4)))
        assert _core.exported_program_to_ir(exported_program)


if __name__ == '__main__':
    unittest.main()
