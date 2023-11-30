from typing import List, Optional
from onnx.defs import OpSchema
from ...values import ParamSchema, TypeConstraint


class NotImplementedFunction:
    def __init__(
        self,
        name: str,
        n_inputs: int,
        n_outputs: int = 1,
        domain: Optional[str] = None,
        opset: int = 1,
    ):
        prefix = name.split("::", maxsplit=1)[0]
        if prefix not in {"aten", "_operator", "internal", "math", "prims"}:
            raise RuntimeError(f"Unexpected name={name!r}.")
        self.prefix_ = prefix
        self.name_ = name
        if domain is None:
            default_values = {"aten": "pkg.onnxscript.torch_lib"}
            self.domain_ = default_values[prefix]
        else:
            self.domain_ = domain
        self.opset_ = opset
        self.param_schemas_ = [
            ParamSchema(name=f"i{i}", is_input=True) for i in range(n_inputs)
        ]

        formal_inputs = [
            OpSchema.FormalParameter(
                f"i{i}",
                f"TI{i}",
                param_option=OpSchema.FormalParameterOption.Single,
                is_homogeneous=False,
            )
            for i in range(n_inputs)
        ]
        formal_outputs = [
            OpSchema.FormalParameter(
                f"o{i}",
                f"TO{i}",
                param_option=OpSchema.FormalParameterOption.Single,
                is_homogeneous=False,
            )
            for i in range(n_outputs)
        ]
        tensor_types = [
            "tensor(float)",
            "tensor(double)",
            "tensor(float16)",
            "tensor(int64)",
            "tensor(int32)",
        ]
        constraints = []
        for i in range(n_inputs):
            tc = TypeConstraint(name=f"TI{i}", allowed_types=tensor_types)
            constraints.append(tc)
        for i in range(n_outputs):
            tc = TypeConstraint(name=f"TO{i}", allowed_types=tensor_types)
            constraints.append(tc)
        self.op_schema_ = OpSchema(
            self.name_,
            self.domain_,
            since_version=self.opset_,
            inputs=formal_inputs,
            outputs=formal_outputs,
            type_constraints=[c.as_tuple() for c in constraints],
            attributes=[],  #  onnx.defs.OpSchema.Attribute(... ]
        )

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            f"Function '{self.__class__.__name__}.{self.name_}' is not implemented. "
            f"args={args}, kwargs={kwargs}."
        )

    @property
    def name(self) -> str:
        return f"{self.prefix_}_{self.name_}"

    @property
    def opset(self) -> int:
        return self.opset_

    @property
    def op_schema(self) -> int:
        return self.op_schema_

    def param_schemas(self) -> List[ParamSchema]:
        return self.param_schemas_


def register_aten_functions(registry=None):
    if registry is None:
        from .registration import default_registry

        registry = default_registry

    dummies = [
        ("aten::add.Tensor", 2),
        ("aten::alias", 1),
        ("aten::convolution", 3),
        ("aten::getitem", 2),
        ("aten::max_pool2d_with_indices", 2),
        ("aten::mm", 2),
        ("aten::mul", 2),
        ("aten::relu", 1),
        ("aten::scalar_tensor", 1),
        ("aten::t", 1),
        ("aten::view", 1),
    ]
    for name, n_inputs in dummies:
        registry.register(
            NotImplementedFunction(name, n_inputs=n_inputs),
            name,
            private=False,
            complex=False,
        )
