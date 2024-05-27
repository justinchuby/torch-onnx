import onnx
import torch
import torch_onnx
from onnxscript import ir
from transformers import LongformerModel, LongformerTokenizer

tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors="pt")
exported = torch.export.export(
    model, (encoded_input["input_ids"], encoded_input["attention_mask"])
)
ir_model = torch_onnx.exported_program_to_ir(exported)
proto = ir.to_proto(ir_model)
onnx.save_model(proto, "longformer.onnx")
