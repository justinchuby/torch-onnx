import torch
from transformers import LongformerModel, LongformerTokenizer

import torch_onnx

torch_onnx.patch_torch()

tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors="pt")
print("Exporting model...")

ep = torch.export.export(
    model,
    (encoded_input["input_ids"], encoded_input["attention_mask"]),
)

results = tuple(
    torch_onnx.verification.minimize_inaccurate_subgraph(ep, atol=1e-4, rtol=1e-2)
)

for i, result in enumerate(results):
    print(f"------------------Graph {i}------------------")
    print(result.graph)
