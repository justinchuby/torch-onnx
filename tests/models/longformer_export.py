import torch
import torch_onnx
from transformers import LongformerModel, LongformerTokenizer

torch_onnx.patch_torch(
    report=True, profile=True, verify=True, dump_exported_program=True
)

tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors="pt")
print("Exporting model...")

torch.onnx.export(
    model,
    (encoded_input["input_ids"], encoded_input["attention_mask"]),
    "longformer.onnx",
    opset_version=18,
)
