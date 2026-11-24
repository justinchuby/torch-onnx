# Copyright: PyTorch contributors. See LICENSE-PYTORCH.
"""Runtime models from test_models_onnxruntime.py at e2c2673d00500d268fcb1cc63fa7cc3173876505.

Removed in 6dca9f63bc4975ac323044a440c2c7e5f8512408. ShuffleNet captures
batch3 to avoid zero/one specialization and still executes original batches1/3.
"""

import torch
import torchvision
from test_runtime_regressions import RecoveredRuntimeHarness
from torch import nn


class TestRecoveredRuntimeModels(RecoveredRuntimeHarness):
    def test_transformer_encoder(self):
        class MyModule(torch.nn.Module):
            def __init__(self, ninp, nhead, nhid, dropout, nlayers):
                super().__init__()
                encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
                self.transformer_encoder = nn.TransformerEncoder(
                    encoder_layers, nlayers
                )

            def forward(self, input):
                return self.transformer_encoder(input)

        x = torch.rand(10, 32, 512)
        self.run_test(MyModule(512, 8, 2048, 0.0, 3), (x,), atol=1e-5)

    def test_mobilenet_v3(self):
        model = torchvision.models.quantization.mobilenet_v3_large(weights=None)
        dummy_input = torch.randn(1, 3, 224, 224)
        self.run_test(model, (dummy_input,))

    def test_shufflenet_v2_dynamic_axes(self):
        model = torchvision.models.shufflenet_v2_x0_5(weights=None)
        dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True)
        test_inputs = torch.randn(3, 3, 224, 224, requires_grad=True)
        self.run_test(
            model,
            (test_inputs,),
            additional_test_inputs=[(dummy_input,), (test_inputs,)],
            input_names=["input_images"],
            output_names=["outputs"],
            dynamic_axes={
                "input_images": {0: "batch_size"},
                "outputs": {0: "batch_size"},
            },
            rtol=1e-3,
            atol=1e-5,
        )
