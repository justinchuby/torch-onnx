# Copyright: PyTorch contributors. See LICENSE-PYTORCH.
"""Factory, embedding and container regressions from PyTorch's deleted suite.

Parent e2c2673d00500d268fcb1cc63fa7cc3173876505, removed by
6dca9f63bc4975ac323044a440c2c7e5f8512408. Original bodies/comments are
retained; regular modules/functions replace the retired scripted frontend.
"""

import torch
from test_runtime_regressions import RecoveredRuntimeHarness
from torch import Tensor


class TestRecoveredFactoryEmbedding(RecoveredRuntimeHarness):
    def test_gather_constant_fold(self):
        class GatherModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Buffer(torch.ones(5))
                # torch.nn.Embedding is converted to ONNX::Gather.
                # Constant folding will be triggered for constant inputs.
                # This pattern is common for constant mask inputs in transformer models.
                self.embed = torch.nn.Embedding(8, 3)

            def forward(self, x):
                # shape is of rank 0
                shape = self.weight.shape[0]
                m = 5 - shape
                y = torch.ones(1, 4, dtype=torch.long)
                return x.clamp(min=m), self.embed(y)

        x = torch.randn(1)
        self.run_test(GatherModule(), (x,))

        class GatherModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Buffer(torch.ones(2))

            def forward(self, x):
                # shape is of rank 0
                shape = self.weight.shape[0]
                pad = [1, shape, shape, shape]
                zero_pad = torch.nn.ZeroPad2d(pad)
                return zero_pad(x)

        x = torch.randn(1, 3, 2)
        self.run_test(GatherModule(), (x,))

        class GatherModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rb = torch.nn.Buffer(torch.randn(1, 1, 3, 1, 1))

            def forward(self, x):
                x += self.rb[0]
                return x

        x = torch.randn(1, 3, 224, 224)
        self.run_test(
            GatherModule(),
            (x,),
            dynamic_axes={
                "input": {0: "batch", 2: "height", 3: "width"},
                "output": {0: "batch", 1: "class", 2: "height", 3: "width"},
            },
            input_names=["input"],
            output_names=["output"],
            additional_test_inputs=[torch.randn(1, 3, 111, 97)],
        )

    def test_embedding_module(self):
        class EmbedModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.emb = torch.nn.Embedding(4, 3, padding_idx=1)
                self.emb2 = torch.nn.Embedding(4, 3, padding_idx=1)
                with torch.no_grad():
                    self.emb2.weight[1] = torch.ones(3)

            def forward(self, input):
                return self.emb(input), self.emb2(input)

        model = EmbedModel()
        x = torch.randint(4, (4,))
        x[2] = x[0] = 1
        self.run_test(model, (x,))

        x = torch.randint(4, (4, 3, 2))
        x[2] = 1
        x[0][1] = 1
        self.run_test(model, (x,))

        class EmbedModelWithoutPaddingIdx(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.emb = torch.nn.Embedding(4, 3)

            def forward(self, input):
                return self.emb(input)

        model = EmbedModelWithoutPaddingIdx()
        x = torch.randint(4, (4, 3, 2))
        self.run_test(model, (x,))

    def test_hann_window_periodic(self):
        class HannWindowModule_Periodic(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.window_length = 0

            def forward(self, x, window_length: int):
                self.window_length = window_length
                return torch.add(
                    x,
                    torch.hann_window(
                        self.window_length, periodic=True, dtype=torch.float
                    ),
                )

        win_length = 100
        x = torch.randn(win_length)

        module = HannWindowModule_Periodic()
        self.run_test(module, (x, win_length))

    def test_hann_window_not_periodic(self):
        class HannWindowModule_NotPeriodic(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.window_length = 0

            def forward(self, x, window_length: int):
                self.window_length = window_length
                return torch.add(
                    x,
                    torch.hann_window(
                        self.window_length, periodic=False, dtype=torch.float
                    ),
                )

        win_length = 100
        x = torch.randn(win_length)

        module = HannWindowModule_NotPeriodic()
        self.run_test(module, (x, win_length))

    def test_hann_window_default_values(self):
        class HannWindowModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.window_length = 0

            def forward(self, x, window_length: int):
                import torch.nn.functional as F

                self.window_length = window_length
                return torch.add(x, F.relu(torch.hann_window(self.window_length)))

        win_length = 100
        x = torch.randn(win_length, dtype=torch.float)
        module = HannWindowModule()

        module(x, win_length)
        self.run_test(module, (x, win_length))

    def test_softmax(self):
        for i in range(-4, 3):
            model = torch.nn.Softmax(dim=i)
            input = torch.randn(3, 4, 5, 6)
            self.run_test(model, input)

            class SoftmaxUnknownRank(torch.nn.Module):
                def __init__(self, i):
                    super().__init__()
                    self.softmax = torch.nn.Softmax(dim=i)

                def forward(self, x):
                    return self.softmax(x.reshape(3, 4, 5, 6))

            model = SoftmaxUnknownRank(i)
            self.run_test(model, input)

    def test_softmax_large_values(self):
        input = torch.tensor(
            [[-1e12, -1e12, -1e12], [1e12, 0.0, -5.0], [3.0, 4.0, 5.0]]
        )
        for i in range(-2, 1):
            model = torch.nn.Softmax(dim=i)
            self.run_test(model, input)

            class SoftmaxUnknownRank(torch.nn.Module):
                def __init__(self, i):
                    super().__init__()
                    self.softmax = torch.nn.Softmax(dim=i)

                def forward(self, x):
                    return self.softmax(x.reshape(3, 3))

            model = SoftmaxUnknownRank(i)
            self.run_test(model, input)

    def test_scatter_add(self):
        class ScatterModel(torch.nn.Module):
            def forward(self, input, indices, values):
                return input.scatter_add(1, indices, values)

        input = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        indices = torch.tensor([[1, 0], [0, 1], [0, 1]], dtype=torch.int64)
        values = torch.tensor([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        self.run_test(ScatterModel(), input_args=(input, indices, values))

        def scatter_sum(src: Tensor, index: Tensor):
            size = src.size()
            out = torch.zeros(size, dtype=src.dtype)
            return out.scatter_add_(1, index, src)

        class ScatterModel(torch.nn.Module):
            def forward(self, src, index):
                return scatter_sum(src, index)

        src = torch.rand(3, 2)
        index = torch.tensor([[0, 1], [0, 1], [0, 1]], dtype=torch.int64)
        self.run_test(ScatterModel(), (src, index))

    def test_scatter_add_index_not_unique(self):
        class ScatterModel(torch.nn.Module):
            def forward(self, input, indices, values):
                return input.scatter_add(1, indices, values)

        input = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        indices = torch.tensor([[0, 0], [1, 1], [2, 2]], dtype=torch.int64)
        values = torch.tensor([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        self.run_test(ScatterModel(), input_args=(input, indices, values))

        def scatter_sum(src: Tensor, index: Tensor):
            size = src.size()
            out = torch.zeros(size, dtype=src.dtype)
            return out.scatter_add_(1, index, src)

        class ScatterModel(torch.nn.Module):
            def forward(self, src, index):
                return scatter_sum(src, index)

        src = torch.rand(3, 2)
        index = torch.tensor([[0, 0], [1, 1], [0, 1]], dtype=torch.int64)
        self.run_test(ScatterModel(), (src, index))

    def test_list_unpack_scripted(self):
        class ListUnpack(torch.nn.Module):
            def forward(self, x):
                a, b = x.shape
                return x.new_zeros((a, b))

        x = torch.randn(2, 3)
        self.run_test(
            ListUnpack(),
            x,
            input_names=["x"],
            dynamic_axes={"x": [0, 1]},
            additional_test_inputs=[torch.randn(7, 9)],
        )
        self.run_test(ListUnpack(), x, remained_onnx_input_idx=[])

    def test_list_unpack_scripted_runs_without_error_with_constructed_list_as_input(
        self,
    ):
        class PackUnpack(torch.nn.Module):
            """Create and unpack a list of tensors.

            When scripted, it should produce a graph similar to

            ```
            graph(%self : __torch__.PackUnpack,
                %a.1 : Tensor,
                %b.1 : Tensor):
            %packed.1 : Tensor[] = prim::ListConstruct(%a.1, %b.1)
            %c.1 : Tensor, %8 : Tensor = prim::ListUnpack(%packed.1)
            return (%c.1)
            ```
            """

            def forward(self, a, b):
                packed = [a, b]
                c, _ = packed
                return c

        self.run_test(
            PackUnpack(),
            (torch.tensor(0), torch.tensor([42])),
            remained_onnx_input_idx=[0],
        )

    def test_list_unpack_slice_scripted(self):
        class ListUnpackSlice(torch.nn.Module):
            def forward(self, x):
                a, b = x.shape[2:]
                return x.new_zeros((a, b))

        x = torch.randn(2, 3, 4, 5)
        self.run_test(
            ListUnpackSlice(),
            x,
            input_names=["x"],
            dynamic_axes={"x": [0, 1, 2, 3]},
            additional_test_inputs=[torch.randn(5, 6, 7, 9)],
        )
        self.run_test(ListUnpackSlice(), x, remained_onnx_input_idx=[])
