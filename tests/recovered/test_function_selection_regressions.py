# Copyright: PyTorch contributors. See LICENSE-PYTORCH.
"""All four local-function selections from test_utility_funs.py:801.

Source: e2c2673d00500d268fcb1cc63fa7cc3173876505. Explicit native functions
replace export_modules_as_functions; automatic module extraction is not claimed.
"""

import onnx_ir as ir
import pytest
import torch
from test_function_scope_regressions import _check_roundtrip_and_optimize

import torch_onnx


class _N(torch.nn.Module):
    def __init__(self, prob):
        super().__init__()
        self.dropout = torch.nn.Dropout(prob)

    def forward(self, x):
        return self.dropout(x)


class _M(torch.nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.lns = torch.nn.ModuleList(
            [torch.nn.LayerNorm(3, eps=i) for i in range(num_layers)]
        )
        self.celu1 = torch.nn.CELU(1.0)
        self.celu2 = torch.nn.CELU(2.0)
        self.dropout = _N(0.5)

    def forward(self, x, y, z):
        res1 = self.celu1(x)
        res2 = self.celu2(y)
        for ln in self.lns:
            z = ln(z)
        return res1 + res2, self.dropout(z)


class _NativeM(_M):
    def forward(self, x, y, z):
        return torch.ops.recovered_function_selection.model.default(
            x,
            y,
            z,
            [ln.weight for ln in self.lns],
            [ln.bias for ln in self.lns],
        )


def _function(domain, name, input_names, attribute_types, build):
    formals = [
        ir.Value(name=name, type=ir.TensorType(ir.DataType.FLOAT))
        for name in input_names
    ]
    graph = ir.Graph(formals, [], nodes=[], name=name, opset_imports={"": 20})
    builder = torch_onnx.GraphBuilder(graph, opset=20)
    graph.outputs.extend(build(builder, formals))
    function = ir.Function(
        domain,
        name,
        "",
        graph=graph,
        attributes={
            name: ir.Attr(name, attribute_type, None)
            for name, attribute_type in attribute_types.items()
        },
    )
    return function, builder.functions


def _make_translation(selection):
    def translation(op, x, y, z, scales, biases):
        celu, _ = _function(
            "torch.nn.modules.activation",
            "CELU",
            ["input"],
            {
                "alpha": ir.AttributeType.FLOAT,
                "inplace": ir.AttributeType.INT,
                "training": ir.AttributeType.INT,
            },
            lambda builder, inputs: builder.call(
                "Celu",
                inputs[0],
                alpha=ir.RefAttr("alpha", "alpha", ir.AttributeType.FLOAT),
            ),
        )
        layer_norm, _ = _function(
            "torch.nn.modules.normalization",
            "LayerNorm",
            ["input", "scale", "bias"],
            {
                "eps": ir.AttributeType.FLOAT,
                "elementwise_affine": ir.AttributeType.INT,
                "training": ir.AttributeType.INT,
            },
            lambda builder, inputs: builder.call(
                "LayerNormalization",
                *inputs,
                axis=-1,
                epsilon=ir.RefAttr("epsilon", "eps", ir.AttributeType.FLOAT),
                _num_outputs=1,
            ),
        )

        def body(builder, inputs):
            context = torch_onnx.TranslationContext(builder)
            results = []
            for value, alpha in zip(inputs[:2], (1.0, 2.0), strict=True):
                if selection == "none":
                    result = context.Celu(value, alpha=alpha)
                else:
                    result = builder.call_function(
                        celu,
                        value,
                        version=1,
                        attributes={
                            "alpha": alpha,
                            "inplace": False,
                            "training": False,
                        },
                    )[0]
                results.append(result)
            normalized = inputs[2]
            for i in range(3):
                if selection in {"all", "celu_layernorm"}:
                    normalized = builder.call_function(
                        layer_norm,
                        normalized,
                        inputs[3 + i],
                        inputs[6 + i],
                        version=1,
                        attributes={
                            "eps": float(i),
                            "elementwise_affine": True,
                            "training": False,
                        },
                    )[0]
                else:
                    normalized = builder.call(
                        "LayerNormalization",
                        normalized,
                        inputs[3 + i],
                        inputs[6 + i],
                        axis=-1,
                        epsilon=float(i),
                        _num_outputs=1,
                    )[0]
            # Inference removes dropout, so N/Dropout must not become functions.
            return context.Add(*results), normalized

        inputs = [x, y, z, *scales, *biases]
        if selection != "all":
            return body(op.builder, inputs)
        module, dependencies = _function(
            "RecoveredFunctionSelection",
            "M",
            [
                "x",
                "y",
                "z",
                *(f"scale_{i}" for i in range(3)),
                *(f"bias_{i}" for i in range(3)),
            ],
            {
                "num_layers": ir.AttributeType.INT,
                "training": ir.AttributeType.INT,
            },
            body,
        )
        op.builder.functions.update(dependencies)
        return tuple(
            op.builder.call_function(
                module,
                *inputs,
                version=1,
                attributes={"num_layers": 3, "training": False},
            )
        )

    return translation


@pytest.mark.parametrize("selection", ["celu_layernorm", "celu", "none", "all"])
@torch.no_grad()
def test_local_function(selection):
    def implementation(x, y, z, scales, biases):
        state = {}
        for i in range(3):
            state[f"lns.{i}.weight"] = scales[i]
            state[f"lns.{i}.bias"] = biases[i]
        return torch.func.functional_call(_M(3).eval(), state, (x, y, z))

    def fake(x, y, z, scales, biases):
        return torch.empty_like(x), torch.empty_like(z)

    with torch.library._scoped_library(
        "recovered_function_selection", "FRAGMENT"
    ) as library:
        name = "recovered_function_selection::model"
        torch.library.define(
            name,
            "(Tensor x, Tensor y, Tensor z, Tensor[] scales, Tensor[] biases) -> (Tensor, Tensor)",
            lib=library,
        )
        torch.library.impl(name, "CompositeExplicitAutograd", lib=library)(
            implementation
        )
        torch.library.register_fake(name, lib=library)(fake)
        target = torch.ops.recovered_function_selection.model.default
        torch_onnx.register_custom_translation(target, _make_translation(selection))
        program = None
        try:
            original, native = _M(3).eval(), _NativeM(3).eval()
            native.load_state_dict(original.state_dict())
            args = tuple(torch.randn(2, 3) for _ in range(3))
            rows = torch.export.Dim("rows", min=2)
            program = torch_onnx.export(
                native,
                args,
                optimize=False,
                verbose=False,
                dynamic_shapes=({0: rows}, {0: rows}, {0: rows}),
            )
            functions = {fn.name: fn for fn in program.model_proto.functions}
            expected = {
                "celu_layernorm": {"CELU", "LayerNorm"},
                "celu": {"CELU"},
                "none": set(),
                "all": {"M", "CELU", "LayerNorm"},
            }[selection]
            assert functions.keys() == expected
            for name, domain, attributes, count in [
                (
                    "CELU",
                    "torch.nn.modules.activation",
                    {"alpha", "inplace", "training"},
                    2,
                ),
                (
                    "LayerNorm",
                    "torch.nn.modules.normalization",
                    {"eps", "elementwise_affine", "training"},
                    3,
                ),
            ]:
                if name not in expected:
                    continue
                assert functions[name].domain == domain
                assert set(functions[name].attribute) == attributes
                nodes = (
                    functions["M"].node
                    if selection == "all"
                    else program.model_proto.graph.node
                )
                calls = [node for node in nodes if node.op_type == name]
                assert len(calls) == count
                for node in calls:
                    assert node.domain == domain
                    assert {attr.name for attr in node.attribute} == attributes
            _check_roundtrip_and_optimize(
                program,
                original,
                [
                    args,
                    tuple(-value for value in args),
                    tuple(torch.randn(5, 3) for _ in range(3)),
                ],
            )
        finally:
            if program is not None:
                program.release()
            torch_onnx.unregister_custom_translation(target)
