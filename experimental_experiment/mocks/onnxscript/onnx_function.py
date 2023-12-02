import inspect
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from onnx.defs import OpSchema
from onnx import AttributeProto, ModelProto
from .values import ParamSchema, TypeConstraint

TENSOR = "TENSOR"


class BaseOnnxFunction:
    def __init__(
        self,
        name: str,
        n_inputs: Union[int, List[str]],
        n_outputs: Union[int, List[str]] = 1,
        domain: Optional[str] = None,
        opset: int = 1,
        register_name: Optional[str] = None,
        kwargs: Dict[str, Any] = None,
    ):
        self.name_ = name
        self.register_name_ = register_name or name
        if domain is None:
            raise ValueError(f"domain cannot be empty for function {name!r}.")
        self.domain_ = domain
        self.opset_ = opset

        if isinstance(n_inputs, int):
            input_names = [f"i{i}" for i in range(n_inputs)]
        else:
            input_names = n_inputs
            n_inputs = len(input_names)

        if isinstance(n_outputs, int):
            output_names = [f"o{i}" for i in range(n_outputs)]
        else:
            input_names = n_outputs
            n_outputs = len(output_names)

        self.param_schemas_ = [
            ParamSchema(name=input_names[i], is_input=True) for i in range(n_inputs)
        ]
        formal_inputs = [
            OpSchema.FormalParameter(
                input_names[i],
                f"TI{i}",
                param_option=OpSchema.FormalParameterOption.Single,
                is_homogeneous=False,
            )
            for i in range(n_inputs)
        ]
        formal_outputs = [
            OpSchema.FormalParameter(
                output_names[i],
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

        def _get_type(v):
            if isinstance(v, int):
                return OpSchema.AttrType(AttributeProto.INT)
            if isinstance(v, float):
                return OpSchema.AttrType(AttributeProto.INT)
            if isinstance(v, tuple):
                if all(map(lambda a: isinstance(a, int), v)):
                    return OpSchema.AttrType(AttributeProto.INTS)
            raise ValueError(f"Unable to guess attribute type for {type(v)}: {v}")

        attributes = []
        if kwargs:
            for k, v in kwargs.items():
                att = OpSchema.Attribute(k, type=_get_type(v))
                attributes.append(att)
                self.param_schemas_.append(ParamSchema(name=k, is_input=False))

        self.op_schema_ = OpSchema(
            self.name_,
            self.domain_,
            since_version=self.opset_,
            inputs=formal_inputs,
            outputs=formal_outputs,
            type_constraints=[c.as_tuple() for c in constraints],
            attributes=attributes,
        )

        self.input_names = input_names
        self.output_names = output_names

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name_!r}, domain={self.domain_!r})"

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            f"Function '{self.__class__.__name__}.{self.name_}' is not implemented. "
            f"args={args}, kwargs={kwargs}."
        )

    @property
    def name(self) -> str:
        return self.name_

    @property
    def register_name(self) -> str:
        return self.register_name_

    @property
    def opset(self) -> int:
        return self.opset_

    @property
    def domain(self) -> int:
        return self.domain_

    @property
    def op_schema(self) -> int:
        return self.op_schema_

    def param_schemas(self) -> List[ParamSchema]:
        return self.param_schemas_

    def get_builder(self):
        from .onnx_opset import default_opset
        from .graph_builder import GraphBuilder

        return GraphBuilder([default_opset])


class OnnxFunction(BaseOnnxFunction):
    def __init__(self, fct: Callable, domain: str, opset: int = 1):
        name = fct.__name__
        if fct.__name__.startswith("aten_"):
            register_name = f"aten::{fct.__name__[5:]}"
        else:
            raise NotImplementedError(
                f"Unable to guess the namespace of function {fct}."
            )

        sig = inspect.signature(fct)
        n_inputs = []
        kwargs = {}
        for i, s in enumerate(sig.parameters):
            p = sig.parameters[s]
            if i == 0:
                if p.annotation == "GraphBuilder" and s == "g":
                    continue
                raise RuntimeError(
                    f"The function {fct} should take an opset as the first parameter "
                    f"the first name is {s!r} ({type(s)}) and its annotation is "
                    f"{p.annotation!r} ({type(p.annotation)})."
                )
            if p.annotation == TENSOR:
                n_inputs.append(s)
            elif p.annotation in (bool, int, Sequence[int]):
                kwargs[p.name] = p.default
            else:
                raise ValueError(
                    f"Unexpected annotation {p.annotation!r} for parameter {p.name!r} and function {fct}."
                )
            p = sig.parameters[s]

        if sig.return_annotation == TENSOR:
            n_outputs = 1
        else:
            raise ValueError(
                f"Unexpected return annotation {sig.return_annotation!r} for function {fct}."
            )

        super().__init__(
            name=name,
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            domain=domain,
            opset=opset,
            kwargs=kwargs,
            register_name=register_name,
        )
        self.fct_ = fct

    def to_onnx(self, *args, **kwargs) -> ModelProto:
        g = self.get_builder()
        output_names = self.fct_(g, *args, **kwargs)
        g.make_output(output_names)
        return g.to_onnx()

    def __call__(self, *args, **kwargs):
        if len(args) == 0:
            from .evaluator import default

            return default().eval_function(self, args, kwargs)

        evaluator = args[0].get_evaluator()
        return evaluator.eval_function(self, args, kwargs)


class TracedOnnxFunction:
    pass
