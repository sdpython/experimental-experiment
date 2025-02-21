from typing import Any, Dict, List
import onnx
from onnx_array_api.translate_api.translate import Translater
from onnx_array_api.translate_api.builder_emitter import BuilderEmitter


class CustomBuilderEmitter(BuilderEmitter):
    """
    Custom :class:`onnx_array_api.translate_api.builder_emitter.BuilderEmitter`.
    """

    def __init__(self, make_model_function: str = "make_my_model"):
        super().__init__(make_model_function="make_my_model")

    def _emit_node_type(self, op_type, op_domain):
        if op_type in {"Squeeze", "Unsqueeze"} or op_type.startswith("Reduce"):
            return f"{op_type}AnyOpset"
        return op_type

    def _clean_result_name(self, name):
        return name.replace("#", "__").replace("-", "_")

    def _emit_end_function(self, **kwargs: Dict[str, Any]) -> List[str]:
        rows = super()._emit_end_function(**kwargs)
        return [
            *rows[:-1],
            "    opts = FunctionOptions(",
            f"        name={self.f_name!r},",
            f"        domain={self.f_domain!r},",
            "        move_initializer_to_constant=True,",
            "    )",
            "    g.make_local_function(gr, opts, optimize=False)",
        ]


def to_graph_builder_code(proto: onnx.ModelProto, function_name: str = "build_model") -> str:
    """
    Produces a code building a model with
    :class:`experimental_experiment.xbuilder.GraphBuilder`.

    :param proto: model to convert into a code
    :param function_name: function name
    :return: str
    """
    tr = Translater(proto, emitter=CustomBuilderEmitter())
    code = tr.export(as_str=True)
    return "\n".join(
        [
            "import numpy as np",
            "from onnx import TensorProto",
            "from onnx.numpy_helper import from_array",
            "from experimental_experiment.xbuilder import GraphBuilder, FunctionOptions",
            "",
            "",
            code.replace("array(nan", "array(np.nan"),
        ]
    )
