from typing import Any, Dict, List, Optional, Union
from ...onnx_function import BaseOnnxFunction, OnnxFunction


class NotImplementedFunction(BaseOnnxFunction):
    def __init__(
        self,
        name: str,
        n_inputs: Union[int, List[str]],
        n_outputs: Union[int, List[str]] = 1,
        domain: Optional[str] = None,
        opset: int = 1,
        kwargs: Dict[str, Any] = None,
    ):
        prefix, short = name.split("::", maxsplit=1)
        if prefix not in {"aten", "_operator", "internal", "math", "prims"}:
            raise RuntimeError(f"Unexpected name={name!r}.")
        if domain is None:
            default_values = {"aten": "pkg.onnxscript.torch_lib"}
            domain = default_values[prefix]
        super().__init__(
            f"{prefix}_{short}",
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            domain=domain,
            opset=opset,
            kwargs=kwargs,
            register_name=name,
        )


def register_aten_functions(registry=None):
    if registry is None:
        from .registration import default_registry

        registry = default_registry

    from ._aten_functions import aten_convolution

    aten_domain = "pkg.onnxscript.torch_lib"
    implemented = [
        aten_convolution,
    ]
    for f in implemented:
        of = OnnxFunction(f, domain=aten_domain)
        registry.register(of, of.register_name, private=False, complex=False)

    dummies = [
        ("aten::add.Tensor", 2),
        ("aten::alias", 1),
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
        of = NotImplementedFunction(name, n_inputs=n_inputs)
        registry.register(of, of.register_name, private=False, complex=False)
