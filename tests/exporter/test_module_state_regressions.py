# Copyright: PyTorch contributors. See LICENSE-PYTORCH.
"""Parameter-field state cases from PyTorch's deleted ONNX runtime suite.

Parent e2c2673d00500d268fcb1cc63fa7cc3173876505; removal
6dca9f63bc4975ac323044a440c2c7e5f8512408. Buffer attributes let eager
references execute the original TorchScript tensor-field assignments.
Persistent bias updates use explicit functional_call state, not implicit
ONNXProgram mutation. Fixed loops/list lengths retain their original bounds.
"""

import copy

import torch
from _common.runtime import RecoveredRuntimeHarness
from torch import Tensor


def _tensor_state_model(model):
    for name, value in tuple(model.conv.named_parameters(recurse=False)):
        delattr(model.conv, name)
        model.conv.register_buffer(name, value.detach().clone())
    return model


class _FunctionalBiasState(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args):
        state = {name: value.clone() for name, value in self.model.named_buffers()}
        state["conv.bias"] = args[-1]
        result = torch.func.functional_call(self.model, state, args[:-1])
        return result, state["conv.bias"]


class TestRecoveredModuleState(RecoveredRuntimeHarness):
    def test_inplace_sequence_with_loop(self):
        class M(torch.nn.Module):
            def __init__(self, tensor_flags):
                super().__init__()
                self.tensor_flags = tensor_flags

            def process(self, beam_hyps: list[Tensor], done, x):
                batch_size = x.shape[0]
                for i in range(batch_size):
                    if done[i]:
                        continue

                    beam_idx = 0
                    for _, token in enumerate(x[i]):
                        beam_hyps.append(token)
                        beam_idx += 1  # noqa: SIM113

                        if beam_idx == 6:
                            break

                    done[i] = len(beam_hyps) > 4

                return beam_hyps, done

            def forward(self, x):
                beam_hyps: list[Tensor] = []
                batch_size = x.shape[0]
                cur_len = 0
                max_len = x.shape[1]
                done = (
                    torch.zeros(batch_size, dtype=torch.bool)
                    if self.tensor_flags
                    else [False] * batch_size
                )
                while cur_len < max_len:
                    beam_hyps, done = self.process(beam_hyps, done, x[:, 0, :])
                    cur_len = cur_len + 1
                return beam_hyps

        x = torch.randn(8, 4, 3)
        reference = M(tensor_flags=True)
        program = self.run_test(M(tensor_flags=False), (x,))
        self.assert_program_roundtrip(
            program,
            examples=[x, -x, torch.zeros_like(x), torch.randn_like(x)],
            reference=reference,
        )

    def run_state_test(self, model, input_args, *, persistent_bias=False):
        args = input_args if isinstance(input_args, tuple) else (input_args,)
        reference = copy.deepcopy(model)
        initial = reference.conv.bias.clone()
        adapter = _FunctionalBiasState(model) if persistent_bias else model
        inputs = (*args, initial) if persistent_bias else args
        program = self.run_test(adapter, inputs)
        reloaded = self.reload_program(program)
        for candidate in (program, reloaded):
            oracle = copy.deepcopy(reference)
            state = initial.clone()
            for values in (
                args,
                tuple(-value for value in args),
                tuple(torch.randn_like(value) for value in args),
            ):
                expected = oracle(*values)
                call_args = (*values, state) if persistent_bias else values
                if persistent_bias:
                    expected = (expected, oracle.conv.bias)
                actual = self.assert_program_outputs(
                    candidate, call_args, expected=expected
                )
                if persistent_bias:
                    state = actual[-1]

    def test_set_attr(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv1d(3, 10, 2)
                self.b = False

            def forward(self, box_regression, weight):
                self.b = True
                self.conv.weight = weight
                w = torch.softmax(self.conv.weight, dim=0)
                self.conv.weight = w + w
                if self.b:
                    return box_regression + self.conv.weight
                else:
                    return box_regression - self.conv.weight

        model = _tensor_state_model(MyModule())
        weight = torch.ones(3, 2)
        box_regression = torch.randn(3, 2)
        self.run_state_test(model, (box_regression, weight))

    def test_set_attr_2(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv1d(10, 3, 3)
                self.conv.bias = torch.nn.Parameter(torch.zeros(3, 10, 3))

            def set_cell_anchors(self, anchors):
                if self.conv.bias is not None:
                    b = self.conv.bias
                    assert b is not None
                    self.conv.bias = anchors + b
                elif self.conv.weight is not None:
                    self.conv.weight = torch.randn(3, 10)
                    self.conv.bias = self.conv.weight[:]

            def forward(self, anchors) -> Tensor | None:
                self.set_cell_anchors(anchors)
                return self.conv.bias

        model = _tensor_state_model(MyModule())
        anchors = torch.ones(3, 10, 3)
        self.run_state_test(model, anchors, persistent_bias=True)

    def test_set_attr_3(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv1d(10, 3, 3)
                self.conv.weight = torch.nn.Parameter(torch.zeros(3, 10))
                self.conv.bias = torch.nn.Parameter(torch.zeros(3, 10, 3))

            def set_cell_anchors(self, anchors, boxes):
                self.conv.weight = torch.ones(3, 10)
                if self.conv.bias is not None:
                    self.conv.bias = torch.randn(3, 10, 3)
                    self.conv.weight = anchors + self.conv.weight
                    boxes[:] = torch.zeros(2, 3)

            def forward(self, anchors) -> tuple[Tensor, Tensor]:
                boxes = torch.ones(2, 2, 3)
                self.set_cell_anchors(anchors, boxes)
                if self.conv.bias is not None:
                    return self.conv.weight, boxes
                return anchors, boxes

        model = _tensor_state_model(MyModule())
        anchors = torch.rand(3, 10)
        self.run_state_test(model, anchors)

    def test_set_attr_4(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv1d(10, 3, 3)
                self.conv.bias = torch.nn.Parameter(torch.zeros(3, 10, 3))

            def set_cell_anchors(self, anchors):
                self.conv.weight = torch.zeros(10, 3)
                if self.conv.bias is not None:
                    w = self.conv.bias
                    assert w is not None
                    self.conv.bias = anchors + w
                else:
                    self.conv.bias = torch.ones(3, 10, 3)

            def forward(self, feature_maps, anchors) -> tuple[Tensor, Tensor]:
                self.set_cell_anchors(anchors)
                result = []
                if self.conv.bias is not None:
                    a = self.conv.bias
                    assert a is not None
                    result += [a]
                result += [feature_maps]
                return result[0], result[1]

        model = _tensor_state_model(MyModule())
        x = torch.rand(5, 11, 30)
        anchors = torch.ones(3, 10, 3)
        self.run_state_test(model, (x, anchors), persistent_bias=True)

    def test_set_attr_5(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv1d(10, 3, 3)
                self.conv.bias = torch.nn.Parameter(torch.zeros(3, 10, 3))

            def set_cell_anchors(self, anchors):
                self.conv.weight = torch.arange(10)
                for i in range(10):
                    if i == 3:
                        for _ in range(10):
                            w = self.conv.weight
                            self.conv.weight = torch.arange(10) + w
                    self.conv.weight = self.conv.weight + torch.arange(10)
                    # NOTE: `is not None` and `assert` is for passing torchscript.
                    if self.conv.bias is not None:
                        a = self.conv.bias
                        assert a is not None
                        self.conv.bias = anchors + a

            def forward(self, anchors):
                self.set_cell_anchors(anchors)
                return self.conv.weight, self.conv.bias

        model = _tensor_state_model(MyModule())
        anchors = torch.ones(3, 10, 3)
        self.run_state_test(model, anchors, persistent_bias=True)

    def test_set_attr_in_loop(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv1d(10, 3, 3)
                self.conv.weight = torch.nn.Parameter(torch.zeros(3, 10))
                self.conv.bias = torch.nn.Parameter(torch.zeros(3, 10, 3))

            def set_cell_anchors(self, anchors, boxes):
                self.conv.weight = torch.randn(3, 10)
                for i in range(self.conv.weight.size(0)):
                    for j in range(10):
                        self.conv.bias = torch.randn(3, 10, 3)
                        self.conv.weight = anchors * i
                        boxes[j] += torch.ones(3, 3)

            def forward(self, anchors) -> tuple[Tensor, Tensor]:
                boxes = torch.ones(10, 3, 3)
                self.set_cell_anchors(anchors, boxes)
                if self.conv.bias is not None:
                    return self.conv.weight, boxes
                return anchors, boxes

        model = _tensor_state_model(MyModule())
        anchors = torch.rand(10)
        self.run_state_test(model, anchors)

    def test_set_attr_in_loop_with_list(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv1d(10, 3, 3)
                self.conv.weight = torch.nn.Parameter(torch.zeros(3, 10))
                self.conv.bias = torch.nn.Parameter(torch.zeros(3, 10, 3))
                self.boxes: list[Tensor] = [
                    torch.ones(1)
                ]  # Workaround placeholder for TorchScript

            def set_cell_anchors(self, anchors):
                self.conv.weight = torch.randn(3, 10)
                for i in range(self.conv.weight.size(0)):
                    for _ in range(10):
                        self.conv.bias = torch.randn(3, 10, 3)
                        self.conv.weight = anchors * i
                        self.boxes.append(torch.ones(3, 3))

            def forward(self, anchors) -> tuple[Tensor, list[Tensor]]:
                self.boxes = []
                self.set_cell_anchors(anchors)
                if self.conv.bias is not None:
                    return self.conv.weight, self.boxes
                return anchors, self.boxes

        model = _tensor_state_model(MyModule())
        anchors = torch.rand(10)
        self.run_state_test(model, anchors)
