import onnx
import torch
import torch_onnx
from onnxscript import ir
from torch import nn


class LlamaMSRotaryEmbedding(nn.Module):
    def __init__(self, hidden_size, num_heads, max_sequence_length):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.max_sequence_length = max_sequence_length

    def get_cos_sin_cache(
        self, theta: float = 10000.0, head_scale=1.0, device="cpu", dtype=torch.float32
    ):
        hidden_size = self.hidden_size
        n_heads = self.num_heads
        max_seq_len = self.max_sequence_length

        # Precalculate rotary matrices for the sequence
        # According to "Attention Is All You Need", theta_i = 10000 ^ (2 * (i - 1)/dim), i in [1, 2, ..., dim//2]
        head_dim = head_scale * hidden_size / n_heads

        pos = torch.arange(0, 2 * (head_dim // 2), step=2, device=device, dtype=dtype)
        freqs = 1.0 / (theta ** (pos / head_dim))

        idx = torch.arange(max_seq_len, device=freqs.device)
        freqs = torch.outer(idx, freqs)

        cos = torch.reshape(torch.cos(freqs), [1, max_seq_len, 1, -1])
        sin = torch.reshape(torch.sin(freqs), [1, max_seq_len, 1, -1])
        dtype = torch.get_default_dtype()

        return cos.to(dtype), sin.to(dtype)

    def forward(self, x, cos, sin, pos):
        # Dimension of x is [batch_size, seq_len, n_heads, head_dim]
        rot_dim = 2 * cos.shape[3]

        # Dolly requires partial rotation
        x_rot = x[:, :, :, :rot_dim]

        x1 = x_rot[:, :, :, 0::2]
        x2 = x_rot[:, :, :, 1::2]

        seq_len = x.shape[1]
        cos_x = cos[:, pos : pos + seq_len, :, :]
        sin_x = sin[:, pos : pos + seq_len, :, :]

        real = cos_x * x1 - sin_x * x2
        imag = sin_x * x1 + cos_x * x2

        x_rot[:, :, :, 0::2] = real
        x_rot[:, :, :, 1::2] = imag

        return torch.cat((x_rot, x[:, :, :, rot_dim:]), dim=-1)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.rotary_embedding_ms = LlamaMSRotaryEmbedding(8, 16, 32)

    def forward(self, x, cos, sin, pos):
        return self.rotary_embedding_ms(x, cos, sin, pos)


batch_size = 2
num_heads = 16
sequence_length = 32
head_size = 8

# Calculated this way to match the data in rotary_embedding_op_test.cc
x_bnsh = torch.randn(batch_size, num_heads, sequence_length, head_size)
x_bsnh = x_bnsh.transpose(1, 2)
rotary_embedding_ms = LlamaMSRotaryEmbedding(head_size, num_heads, sequence_length)
model = Model()
cos_ms, sin_ms = rotary_embedding_ms.get_cos_sin_cache()
pos_ms = 0

lower = "at_conversion"
print("Exporting model...")
exported = torch.export.export(model, (x_bsnh, cos_ms, sin_ms, pos_ms))
print("ExportedProgram done.")
ir_model = torch_onnx.exported_program_to_ir(exported, lower=lower)
proto = ir.to_proto(ir_model)
onnx.save_model(proto, f"rotary_embedding_ms_lower_{lower}.onnx")
