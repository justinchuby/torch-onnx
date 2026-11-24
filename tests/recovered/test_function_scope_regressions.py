# Copyright: PyTorch contributors. See LICENSE-PYTORCH.
"""Utility cases 963/988/1049 from PyTorch e2c2673d00500d268fcb1cc63fa7cc3173876505.

Explicit native registration/functions replace automatic module extraction.
The missing-Embedding-attribute fixture uses ordinary native model export.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import onnx
import onnx_ir as ir
import torch
from torch.utils import _pytree

import torch_onnx


@torch.no_grad()
def _check_roundtrip_and_optimize(program, reference, examples):
    onnx.checker.check_model(program.model_proto, full_check=True)
    with TemporaryDirectory() as directory:
        path = Path(directory) / "function-scopes.onnx"
        program.save(path, external_data=False)
        reloaded = torch_onnx.ONNXProgram(ir.load(path), program.exported_program)
        for candidate in (program, reloaded):
            for args in examples:
                expected = _pytree.tree_flatten(reference(*args))[0]
                torch.testing.assert_close(
                    list(candidate(*args)), expected, rtol=1e-3, atol=1e-7
                )
    program.optimize()
    program.release()
    onnx.checker.check_model(program.model_proto, full_check=True)
    for args in examples:
        torch.testing.assert_close(
            list(program(*args)),
            _pytree.tree_flatten(reference(*args))[0],
            rtol=1e-3,
            atol=1e-7,
        )


def test_local_function_infer_scopes():
    class M(torch.nn.Module):
        def forward(self, x):
            # Concatenation of scalars inserts unscoped tensors in IR graph.
            new_tensor_shape = x.size()[:-1] + (1, 1, -1)
            tensor = x.view(*new_tensor_shape)
            return tensor

    class NativeM(torch.nn.Module):
        def forward(self, x):
            return torch.ops.recovered_scopes.reshape.default(x)

    def implementation(x):
        return M()(x).clone()

    def translation(op: torch_onnx.TranslationContext, x: ir.Value) -> ir.Value:
        formal = ir.Value(
            name="x",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([None, None]),
        )
        body = ir.Graph(
            [formal], [], nodes=[], name="scoped_reshape", opset_imports={"": 20}
        )
        local = torch_onnx.TranslationContext(torch_onnx.GraphBuilder(body, opset=20))
        prefix = local.Slice(local.Shape(formal), [0], [-1], [0])
        target_shape = local.Concat(prefix, [1, 1, -1], axis=0)
        body.outputs.append(local.Reshape(formal, target_shape))
        function = ir.Function("RecoveredScopes", "M", "", graph=body, attributes=[])
        return op.builder.call_function(function, x, version=1)[0]

    with torch.library._scoped_library("recovered_scopes", "FRAGMENT") as library:
        name = "recovered_scopes::reshape"
        torch.library.define(name, "(Tensor x) -> Tensor", lib=library)
        torch.library.impl(name, "CompositeExplicitAutograd", lib=library)(
            implementation
        )
        torch.library.register_fake(name, lib=library)(implementation)
        target = torch.ops.recovered_scopes.reshape.default
        torch_onnx.register_custom_translation(target, translation)
        try:
            x = torch.randn(4, 5)
            program = torch_onnx.export(
                NativeM().eval(),
                (x,),
                verbose=False,
                optimize=False,
                dynamic_shapes=(
                    {
                        0: torch.export.Dim("rows", min=2),
                        1: torch.export.Dim("columns", min=2),
                    },
                ),
            )
            assert [fn.name for fn in program.model_proto.functions] == ["M"]
            function = next(iter(program.model.functions.values()))
            assert any(node.op_type == "Constant" for node in function.graph)
            for node in function.graph:
                for value in node.inputs:
                    if value is not None and value.producer() is not None:
                        assert value.producer().graph is function.graph
            _check_roundtrip_and_optimize(
                program, M(), [(x,), (-x,), (torch.randn(7, 3),)]
            )
        finally:
            torch_onnx.unregister_custom_translation(target)


def test_local_function_predefined_attributes():
    class M(torch.nn.Module):
        num_layers: int

        def __init__(self, num_layers):
            super().__init__()
            self.num_layers = num_layers
            self.lns = torch.nn.ModuleList(
                [torch.nn.LayerNorm(3, eps=1e-4) for _ in range(num_layers)]
            )

        def forward(self, x):
            for ln in self.lns:
                x = ln(x)
            return x

    class NativeM(M):
        def forward(self, x):
            return torch.ops.recovered_attributes.normalize.default(
                x,
                [ln.weight for ln in self.lns],
                [ln.bias for ln in self.lns],
                self.num_layers,
                self.lns[0].eps,
                self.lns[0].elementwise_affine,
            )

    def implementation(x, scales, biases, layers, eps, affine):
        for i in range(layers):
            x = torch.nn.functional.layer_norm(x, (3,), scales[i], biases[i], eps)
        return x

    def fake(x, scales, biases, layers, eps, affine):
        return torch.empty_like(x)

    def translation(
        op: torch_onnx.TranslationContext,
        x: ir.Value,
        scales: list[ir.Value],
        biases: list[ir.Value],
        layers: int,
        eps: float,
        affine: bool,
    ) -> ir.Value:
        tensor_type = ir.TensorType(ir.DataType.FLOAT)
        formals = [
            ir.Value(name=name, type=tensor_type) for name in ("input", "scale", "bias")
        ]
        ln_body = ir.Graph(
            formals, [], nodes=[], name="layer_norm", opset_imports={"": 20}
        )
        local_ln = torch_onnx.GraphBuilder(ln_body, opset=20)
        ln_body.outputs.append(
            local_ln.call(
                "LayerNormalization",
                *formals,
                axis=-1,
                epsilon=ir.RefAttr("epsilon", "eps", ir.AttributeType.FLOAT),
                _num_outputs=1,
            )[0]
        )
        layer_norm = ir.Function(
            "RecoveredAttributes",
            "LayerNorm",
            "",
            graph=ln_body,
            attributes={
                "eps": ir.Attr("eps", ir.AttributeType.FLOAT, None),
                "elementwise_affine": ir.Attr(
                    "elementwise_affine", ir.AttributeType.INT, None
                ),
            },
        )
        formals = [
            ir.Value(name=f"arg_{i}", type=tensor_type) for i in range(1 + 2 * layers)
        ]
        body = ir.Graph(formals, [], nodes=[], name="module", opset_imports={"": 20})
        local_m = torch_onnx.GraphBuilder(body, opset=20)
        value = formals[0]
        for i in range(layers):
            value = local_m.call_function(
                layer_norm,
                value,
                formals[1 + i],
                formals[1 + layers + i],
                version=1,
                attributes={"eps": eps, "elementwise_affine": affine},
            )[0]
        body.outputs.append(value)
        module = ir.Function(
            "RecoveredAttributes",
            "M",
            "",
            graph=body,
            attributes={
                "num_layers": ir.Attr("num_layers", ir.AttributeType.INT, None)
            },
        )
        # Retain functions referenced by the separately built M body.
        op.builder.functions.update(local_m.functions)
        return op.builder.call_function(
            module,
            x,
            *scales,
            *biases,
            version=1,
            attributes={"num_layers": layers},
        )[0]

    with torch.library._scoped_library("recovered_attributes", "FRAGMENT") as library:
        name = "recovered_attributes::normalize"
        torch.library.define(
            name,
            "(Tensor x, Tensor[] scales, Tensor[] biases, int layers, float eps, bool affine) -> Tensor",
            lib=library,
        )
        torch.library.impl(name, "CompositeExplicitAutograd", lib=library)(
            implementation
        )
        torch.library.register_fake(name, lib=library)(fake)
        target = torch.ops.recovered_attributes.normalize.default
        torch_onnx.register_custom_translation(target, translation)
        try:
            generator = torch.Generator().manual_seed(0)
            x = torch.randn(2, 3, generator=generator)
            reference = M(3).eval()
            native = NativeM(3).eval()
            native.load_state_dict(reference.state_dict())
            program = torch_onnx.export(native, (x,), verbose=False, optimize=False)
            functions = {f.name: f for f in program.model_proto.functions}
            assert list(functions["M"].attribute) == ["num_layers"]
            assert list(functions["LayerNorm"].attribute) == [
                "eps",
                "elementwise_affine",
            ]
            node = next(n for n in program.model_proto.graph.node if n.op_type == "M")
            assert node.attribute[0] == onnx.helper.make_attribute("num_layers", 3)
            expected = [
                onnx.helper.make_attribute("eps", reference.lns[0].eps),
                onnx.helper.make_attribute(
                    "elementwise_affine", reference.lns[0].elementwise_affine
                ),
            ]
            assert len(functions["M"].node) == reference.num_layers
            for node in functions["M"].node:
                assert node.op_type == "LayerNorm"
                assert len(node.attribute) == 2
                for attr in node.attribute:
                    assert attr in expected
            _check_roundtrip_and_optimize(
                program,
                reference,
                [(x,), (-x,), (torch.randn(2, 3, generator=generator),)],
            )
        finally:
            torch_onnx.unregister_custom_translation(target)


# This test cases checks the issue where an object does not have an attribute.
# When enabling `export_modules_as_functions = True`, the exporter could return an
# AttributeError. With this test case, we check that the export passes successfully
# without any AttributeError exceptions.
# See https://github.com/pytorch/pytorch/pull/109759 for an example. The exception that
# this test tries to avoid is `AttributeError: 'Embedding' object has no attribute 'freeze'`.
def test_local_function_subset_of_predefined_attributes():
    class M(torch.nn.Module):
        num_layers: int

        def __init__(self, num_layers):
            super().__init__()
            self.embed_layer = torch.nn.Embedding.from_pretrained(
                torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
            )
            self.num_layers = num_layers
            self.lns = torch.nn.ModuleList(
                [torch.nn.LayerNorm(3, eps=1e-4) for _ in range(num_layers)]
            )

        def forward(self, x):
            e = self.embed_layer(torch.LongTensor([1]))
            for ln in self.lns:
                x = ln(x)
            return x, e

    generator = torch.Generator().manual_seed(0)
    x = torch.randn(2, 3, generator=generator)
    model = M(3).eval()
    assert not hasattr(model.embed_layer, "freeze")
    program = torch_onnx.export(model, (x,), verbose=False, optimize=False)
    _check_roundtrip_and_optimize(
        program, model, [(x,), (-x,), (torch.randn(2, 3, generator=generator),)]
    )
