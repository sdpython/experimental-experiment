import inspect
from typing import List
from ...xbuilder import GraphBuilder, FunctionOptions
from ..patterns_api import EasyPatternOptimization


def get_investigation_patterns(verbose: int = 0) -> List["PatternOptimization"]:  # noqa: F821
    """
    Returns a default list of patterns for investigations.
    They do nothing but prints information if verbose > 0.

    .. runpython::
        :showcode:
        :rst:

        from experimental_experiment.xoptim.patterns_api import pattern_table_doc
        from experimental_experiment.xoptim.patterns_investigation import (
            get_investigation_patterns
        )

        print(pattern_table_doc(get_investigation_patterns(), as_rst=True))
    """
    from .element_wise import BinaryInvestigation
    from .llm_patterns import (
        LocalFunctionPackedMatMulPattern,
        LocalFunctionPowTanhPattern,
        LocalFunctionSplitRotaryMulPattern,
    )

    return [
        BinaryInvestigation(verbose=verbose),
        LocalFunctionPackedMatMulPattern(verbose=verbose),
        LocalFunctionPowTanhPattern(verbose=verbose),
        LocalFunctionSplitRotaryMulPattern(verbose=verbose),
    ]


class SimplifyingEasyPatternFunction(EasyPatternOptimization):
    """
    Base class to build investigation patterns.
    See :class:`FunctionPowTanhPattern
    <experimental_experiment.xoptim.patterns_investigation.llm_patterns.FunctionPowTanhPattern>`
    to see how to use it.
    """

    @classmethod
    def f_name(cls) -> str:
        return cls.__name__.replace("Pattern", "").replace("Function", "")

    def post_apply_pattern(self, g, *nodes):
        sig = inspect.signature(self.match_pattern)
        inputs = []
        for pos, p in enumerate(sig.parameters):
            if pos >= 1:
                inputs.append(p)
        domain = "SimplifyingFunction"

        f_name = self.f_name()
        if not g.builder.has_local_function(f_name, domain=domain):
            self._add_local_function(g.builder, domain, f_name, inputs)

    def _add_local_function(
        self, g: GraphBuilder, domain: str, f_name: str, inputs: List[str]
    ):
        local_g = GraphBuilder(g.main_opset, as_function=True)
        local_g.make_tensor_input(inputs)
        last = self.match_pattern(local_g, *inputs)
        local_g.make_tensor_output(last)

        function_options = FunctionOptions(export_as_function=True, name=f_name, domain=domain)
        g.make_local_function(local_g, function_options=function_options)
