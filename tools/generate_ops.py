"""Output ONNX spec in YAML format.

Usage:

    python onnx-spec/tools/spec_to_yaml.py --output onnx-spec/defs
"""

import argparse
import dataclasses
import pathlib
import textwrap

import onnx


@dataclasses.dataclass
class Attribute:
    name: str
    description: str
    type: str
    required: bool
    default_value: str | int | float | list[str] | list[int] | list[float] | None = None


@dataclasses.dataclass
class FormalParameter:
    name: str
    type_str: str
    description: str
    min_arity: int
    tags: list[str]


@dataclasses.dataclass
class TypeConstraintParam:
    type_param_str: str
    description: str
    allowed_type_strs: list[str]


@dataclasses.dataclass
class OpSchema:
    domain: str
    name: str
    since_version: int
    min_input: int
    max_input: int
    min_output: int
    max_output: int
    doc: str
    attributes: list[Attribute]
    inputs: list[FormalParameter]
    outputs: list[FormalParameter]
    type_constraints: list[TypeConstraintParam]
    function: str | None = None
    support_level: str = "COMMON"
    deprecated: bool = False

    def as_dict(self):
        d = dataclasses.asdict(self)
        for attribute in d["attributes"]:
            if attribute["default_value"] is None:
                del attribute["default_value"]
        if not self.function:
            del d["function"]
        return d

    @classmethod
    def from_onnx_opschema(cls, schema: onnx.defs.OpSchema) -> "OpSchema":
        return cls(
            support_level="COMMON"
            if schema.support_level == onnx.defs.OpSchema.SupportType.COMMON
            else "EXPERIMENTAL",
            doc=_process_documentation(schema.doc),
            since_version=schema.since_version,
            deprecated=schema.deprecated,
            domain=schema.domain,
            name=schema.name,
            min_input=schema.min_input,
            max_input=schema.max_input,
            min_output=schema.min_output,
            max_output=schema.max_output,
            attributes=[
                Attribute(
                    name=attr.name,
                    description=_process_documentation(attr.description),
                    type=str(attr.type).split(".")[-1],
                    required=attr.required,
                    default_value=_get_attribute_default_value(attr),
                )
                for attr in schema.attributes.values()
            ],
            inputs=[
                FormalParameter(
                    name=input_.name,
                    type_str=input_.type_str,
                    description=input_.description,
                    min_arity=input_.min_arity,
                    tags=_generate_formal_parameter_tags(input_),
                )
                for input_ in schema.inputs
            ],
            outputs=[
                FormalParameter(
                    name=output.name,
                    type_str=output.type_str,
                    description=output.description,
                    min_arity=output.min_arity,
                    tags=_generate_formal_parameter_tags(output),
                )
                for output in schema.outputs
            ],
            type_constraints=[
                TypeConstraintParam(
                    type_param_str=type_constraint.type_param_str,
                    description=type_constraint.description,
                    allowed_type_strs=list(type_constraint.allowed_type_strs),
                )
                for type_constraint in schema.type_constraints
            ],
        )


def _generate_formal_parameter_tags(
    formal_parameter: onnx.defs.OpSchema.FormalParameter,
) -> list[str]:
    tags: list[str] = []
    if onnx.defs.OpSchema.FormalParameterOption.Optional == formal_parameter.option:
        tags = ["optional"]
    elif onnx.defs.OpSchema.FormalParameterOption.Variadic == formal_parameter.option:
        if formal_parameter.is_homogeneous:
            tags = ["variadic"]
        else:
            tags = ["variadic", "heterogeneous"]

    if (
        onnx.defs.OpSchema.DifferentiationCategory.Differentiable
        == formal_parameter.differentiation_category
    ):
        tags.append("differentiable")
    elif (
        onnx.defs.OpSchema.DifferentiationCategory.NonDifferentiable
        == formal_parameter.differentiation_category
    ):
        tags.append("non-differentiable")

    return tags


def _get_attribute_default_value(attr: onnx.defs.OpSchema.Attribute):
    value = onnx.helper.get_attribute_value(attr.default_value)
    # TODO(justinchuby): Handle when there is no default
    if value is None:
        return None
    if attr.type == onnx.AttributeProto.STRING:
        value = value.decode("utf-8")
    elif attr.type == onnx.AttributeProto.STRINGS:
        value = [v.decode("utf-8") for v in value]
    elif attr.type == onnx.AttributeProto.GRAPH:
        value = onnx.printer.to_text(value)
    elif attr.type == onnx.AttributeProto.GRAPHS:
        value = [onnx.printer.to_text(v) for v in value]
    elif attr.type == onnx.AttributeProto.TENSOR:
        try:
            value = onnx.numpy_helper.to_array(value)
        except Exception:
            value = None
    elif attr.type == onnx.AttributeProto.TENSORS:
        try:
            value = [onnx.numpy_helper.to_array(v) for v in value]
        except Exception:
            value = None

    return value


def _process_documentation(doc: str | None) -> str:
    # Lifted from ONNX's docsgen:
    # https://github.com/onnx/onnx/blob/3fd41d249bb8006935aa0031a332dd945e61b7e5/docs/docsgen/source/onnx_sphinx.py#L414
    if not doc:
        return ""
    doc = textwrap.dedent(doc or "")
    rep = {
        "<dl>": "",
        "</dl>": "",
        "<dt>": "* ",
        "<dd>": "  ",
        "</dt>": "",
        "</dd>": "",
        "<tt>": "`",
        "</tt>": "`",
        "<br>": "\n",
    }
    for k, v in rep.items():
        doc = doc.replace(k, v)
    doc = doc.strip()
    return doc


def build_signature(schema: OpSchema):
    """Build a signature.

    def OpName_1(
        input1: torch.Tensor,
        input2: torch.Tensor,
        output1: torch.Tensor,
        output2: torch.Tensor,
        attr1: int,
        attr2: str,
        attr3: float,
        attr4: bool,
    ) -> torch.Tensor: ...
    """
    inputs = " ".join(
        f"{input_.name}: torch.Tensor," for input_ in schema.inputs
    )
    attributes = " ".join(
        f"{attr.name}: {attr.type.lower()}," for attr in schema.attributes
    )
    return (
        f'''\
def {schema.name}_{schema.since_version}({inputs} {attributes}) -> torch.Tensor: ...
r"""
{schema.doc}
"""
'''
    )

def build_pyi(schemas: list[OpSchema]):
    """Build a .pyi file."""
    signatures = [build_signature(schema) for schema in schemas]
    return "import torch\n\n\n" + "\n\n".join(signatures)


def main():
    parser = argparse.ArgumentParser(description="Output ONNX spec in YAML format.")
    parser.add_argument("--output", help="Output directory", required=True)
    args = parser.parse_args()

    schemas = onnx.defs.get_all_schemas_with_history()

    latest_versions = {}
    for schema in schemas:
        if schema.name in latest_versions:
            latest_versions[schema.name] = max(
                latest_versions[schema.name], schema.since_version
            )
        else:
            latest_versions[schema.name] = schema.since_version
    dataclass_schemas = [OpSchema.from_onnx_opschema(schema) for schema in schemas]
    pyi_file = build_pyi(dataclass_schemas)
    output_dir = pathlib.Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "__init__.pyi", "w") as f:
        f.write(pyi_file)



if __name__ == "__main__":
    main()
