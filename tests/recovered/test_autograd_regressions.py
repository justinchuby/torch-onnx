# Copyright: PyTorch contributors. See LICENSE-PYTORCH.
"""Autograd forward-inlining regressions from PyTorch e2c2673d00500d268fcb1cc63fa7cc3173876505.

Removed in 6dca9f63bc4975ac323044a440c2c7e5f8512408. Modern capture
inlines forward bodies, not legacy symbolic methods or TorchScript fallbacks.
These are inference/export checks, not tests of the original backward formulas.
"""

from __future__ import annotations

import torch
from test_runtime_regressions import RecoveredRuntimeHarness

import torch_onnx


class TestRecoveredAutograd(RecoveredRuntimeHarness):
    def test_auto_grad(self):
        class MyClip(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input, scalar):
                ctx.save_for_backward(input)
                return input.clamp(min=scalar)

        class MyRelu(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                ctx.save_for_backward(input)
                return input.clamp(min=0)

        class MyClipModule(torch.nn.Module):
            def forward(self, x, min):
                return MyClip.apply(x, min)

        x = torch.randn(3, 3)
        min = torch.tensor([0.0])
        self.run_test(
            MyClipModule(),
            (x, min),
            additional_test_inputs=[
                (torch.arange(9.0).reshape(3, 3) - 4, torch.tensor([1.5]))
            ],
        )

        class MyReluModule(torch.nn.Module):
            def forward(self, x):
                return MyRelu.apply(x)

        x = torch.randn(3, 3)
        self.run_test(
            MyReluModule(),
            x,
            additional_test_inputs=[torch.arange(9.0).reshape(3, 3) - 4],
        )

    def test_single_output(self):
        class SingleOut(torch.autograd.Function):
            @staticmethod
            def forward(ctx, i):
                result = i.exp()
                result = result.log()
                ctx.save_for_backward(result)
                return result

            @staticmethod
            def backward(ctx, grad_output):
                (result,) = ctx.saved_tensors
                return grad_output * result

        class Caller(torch.nn.Module):
            def forward(self, input):
                result = input + 5
                return SingleOut.apply(result) + 3

        self.run_test(
            Caller(),
            torch.ones(1),
            additional_test_inputs=[torch.full((1,), 2.0)],
        )

    def test_multi_output(self):
        class MultiOut(torch.autograd.Function):
            @staticmethod
            def forward(ctx, i):
                result_exp = i.exp()
                result_log = result_exp.log()
                ctx.save_for_backward(result_exp, result_log)
                return result_exp, result_log

            @staticmethod
            def backward(ctx, grad_output):
                (result,) = ctx.saved_tensors
                return grad_output * result

        class Caller(torch.nn.Module):
            def forward(self, input):
                return MultiOut.apply(input)

        self.run_test(
            Caller(),
            torch.ones(1, 5),
            additional_test_inputs=[torch.linspace(0.5, 2, 5).reshape(1, 5)],
        )

    def test_partial_output(self):
        class PartialOut(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                ctx.save_for_backward(input)
                values, _ = torch.topk(input, 3)
                return values

        class Caller(torch.nn.Module):
            def forward(self, input):
                return PartialOut.apply(input)

        self.run_test(
            Caller(),
            torch.ones(1, 5),
            additional_test_inputs=[torch.tensor([[2.0, -1.0, 5.0, 3.0, 0.0]])],
        )

    def test_nested_autograd(self):
        class Child(torch.autograd.Function):
            @staticmethod
            def forward(ctx, i):
                result = i.log()
                result_log = result.log()
                ctx.save_for_backward(result_log)
                return result_log

            @staticmethod
            def backward(ctx, grad_output):
                (result,) = ctx.saved_tensors
                return grad_output * result

        class Parent(torch.autograd.Function):
            @staticmethod
            def forward(ctx, i):
                result_exp = i.exp()
                result_log = Child.apply(result_exp)
                ctx.save_for_backward(result_exp, result_log)
                return result_exp, result_log

            @staticmethod
            def backward(ctx, grad_output):
                (result,) = ctx.saved_tensors
                return grad_output * result

        class Caller(torch.nn.Module):
            def forward(self, input):
                return Parent.apply(input)

        self.run_test(
            Caller(),
            torch.ones(1, 5),
            additional_test_inputs=[torch.linspace(0.5, 2, 5).reshape(1, 5)],
        )

    def test_aten_unsupported(self):
        # Original: "Run export in ONNX_FALLTHROUGH mode as torch.erf() is not supported".
        # Erf is now supported: require native ONNX, not retired PythonOp/ATen fallbacks.
        class Erf(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                erf_out = torch.special.erf(x)
                ctx.save_for_backward(erf_out)
                return erf_out

            @staticmethod
            def backward(ctx, grad_output):
                result = ctx.saved_tensors
                return torch.special.erfinv(result), None

        class Caller(torch.nn.Module):
            def forward(self, input):
                return Erf.apply(input)

        model = Caller()
        input = torch.ones(1, 5)
        self.run_test(
            model,
            input,
            additional_test_inputs=[torch.linspace(-2, 2, 5).reshape(1, 5)],
        )
        program = torch_onnx.export(model, (input,), optimize=False)
        self.assertEqual(
            [(node.domain, node.op_type) for node in program.model.graph],
            [("", "Erf")],
        )

    def test_inline_and_symbolic(self):
        class Exp(torch.autograd.Function):
            @staticmethod
            def forward(ctx, i):
                ctx.save_for_backward(i)
                return i.exp()

            @staticmethod
            def symbolic(g, input):
                return g.op("Exp", input)

        class LogLog(torch.autograd.Function):
            @staticmethod
            def forward(ctx, i):
                ctx.save_for_backward(i)
                return i.log().log()

        class Caller(torch.nn.Module):
            def forward(self, input):
                exp_result = Exp.apply(input)
                return LogLog.apply(exp_result)

        model = Caller()
        input = torch.ones(1)
        self.run_test(model, input, additional_test_inputs=[torch.full((1,), 2.0)])
        program = torch_onnx.export(model, (input,), optimize=False)
        self.assertEqual(
            [node.op_type for node in program.model.graph], ["Exp", "Log", "Log"]
        )

    def test_inline_with_scoped_tracing(self):
        class Exp(torch.autograd.Function):
            @staticmethod
            def forward(ctx, i):
                ctx.save_for_backward(i)
                return i.exp()

            @staticmethod
            def symbolic(g, input):
                return g.op("Exp", input)

        class LogLog(torch.autograd.Function):
            @staticmethod
            def forward(ctx, i):
                ctx.save_for_backward(i)
                return i.log().log()

        class Caller(torch.nn.Module):
            def forward(self, input):
                exp_result = Exp.apply(input)
                return LogLog.apply(exp_result)

        model = Caller()
        input = torch.ones(1)
        self.run_test(model, input, additional_test_inputs=[torch.full((1,), 2.0)])
        program = torch_onnx.export(model, (input,), optimize=False)
        self.assertTrue(
            any(
                "Caller" in node.metadata_props.get("namespace", "")
                for node in program.model.graph
            )
        )
