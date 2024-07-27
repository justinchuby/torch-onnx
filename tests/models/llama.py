# Load model directly
import torch
import torch_onnx
import torch_onnx._verification
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
    print("Model loaded")

    kwargs = tokenizer("What is your name?")
    input_ids = torch.tensor([kwargs["input_ids"]])
    attention_mask = torch.tensor([kwargs["attention_mask"]])

    program = torch_onnx.export(
        model, (input_ids, attention_mask), report=True, verify=True
    )
    # program.save("llama31.onnx", include_initializers=False)

    # verification_infos = torch_onnx.verify_onnx_program(program)
    # torch_onnx._verification.save_node_data_for_model_explorer(verification_infos)


if __name__ == "__main__":
    main()
