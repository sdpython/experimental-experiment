from typing import Any, Dict, List, Optional
from ..xbuilder.graph_builder import GraphBuilder
from ._aten_functions import aten__grouped_mm

T = str


def transformers_grouped_mm_fallback(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    mat_a: T,
    mat_b: T,
    offs: Optional[T] = None,
    name: str = "transformers_grouped_mm_fallback",
):
    "transformers_grouped_mm_fallback"
    return aten__grouped_mm(g, sts, outputs, mat_a, mat_b, offs=offs, name=name)
