# Copyright: PyTorch contributors. See LICENSE-PYTORCH.
"""Runtime models recovered from PyTorch e2c2673d00500d268fcb1cc63fa7cc3173876505.

Sources: test_models_onnxruntime.py and test_pytorch_onnx_onnxruntime.py.

Removed in 6dca9f63bc4975ac323044a440c2c7e5f8512408. ShuffleNet captures
batch3 to avoid zero/one specialization and still executes original batches1/3.
"""

import torch
import torchvision
from _common.runtime import RecoveredRuntimeHarness
from torch import nn

import torch_onnx


class TestRecoveredRuntimeModels(RecoveredRuntimeHarness):
    def test_sage_conv(self):
        from torch_geometric import nn as torch_geometric_nn

        class MySAGEConv(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.SAGEConvBlock1 = torch_geometric_nn.SAGEConv(
                    2, 512, normalize=True
                )
                # The source's invalid nn.Seq(Lin(...)) and other unused layers
                # do not participate in this forward path.

            def forward(self, coords0, coords1, edge_from, edge_to):
                adj = torch.cat((edge_from, edge_to), dim=0)
                gra = torch.transpose(torch.cat((coords0, coords1), dim=0), 0, 1)
                x1 = self.SAGEConvBlock1(gra, edge_index=adj)
                x = torch.unsqueeze(torch.sum(x1), dim=0)
                return x

        def inputs_for(nodes):
            coords0 = torch.randn(1, nodes)
            coords1 = torch.randn(1, nodes)
            coords = torch.transpose(torch.cat((coords0, coords1), dim=0), 0, 1)
            # Equivalent loop=True, k=2 topology without a torch-cluster binary
            # dependency; neighbor construction is outside the exported model.
            nearest = torch.cdist(coords, coords).topk(2, largest=False).indices
            edge_from = nearest.reshape(1, -1)
            edge_to = torch.arange(nodes).repeat_interleave(2).reshape(1, -1)
            return coords0, coords1, edge_from, edge_to

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            inputs = inputs_for(6)
            model = MySAGEConv().eval()
            nodes = torch.export.Dim("nodes", min=3)
            program = torch_onnx.export(
                model,
                inputs,
                dynamic_shapes=(
                    {1: nodes},
                    {1: nodes},
                    {1: 2 * nodes},
                    {1: 2 * nodes},
                ),
                input_names=["coords0", "coords1", "edge_from", "edge_to"],
                output_names=["outputs"],
                verbose=False,
            )
            self.assert_program_roundtrip(
                program,
                examples=[inputs, inputs_for(4), inputs_for(8)],
                reference=model,
            )
            program.release()

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
