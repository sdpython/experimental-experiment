from typing import Any, Callable, Dict, List, Optional
from .dispatcher import Dispatcher
from .oxs_opset import OxsOpset, Var


class OxsDispatcher(Dispatcher):
    """
    If class :class:`DynamoInterpreter
    <experimental_experiment.torch_interpreter.interpreter.DynamoInterpreter>`
    cannot find any converting function for a specific function,
    it tries to find an existing one in :epkg:`onnxscript`.
    The converting function from onnxscript is run in trace only mode.
    The variable and functions op, Rank, IsScalar are replaced by
    `op = OwsOpset()`, `op.Rank`, `op.Scalar`.
    onnxscript may have multiple overloaded functions.
    Right now, it takes the first one.

    :param verbose: verbose
    """

    def __init__(self, verbose: int = 0):
        super(OxsDispatcher, self).__init__({}, verbose=verbose)
        self._submodule = None

    @property
    def submodules(self) -> Dict[str, Callable]:
        """
        Returns the submodules implementing torch functions.
        """
        if self._submodule is not None:
            return self._submodule
        from onnxscript.function_libs.torch_lib.ops import (
            core,
            fft,
            linalg,
            nn,
            prims,
            special,
            vision,
        )

        subs = {
            "onnxscript.function_libs.torch_lib.ops.core": core,
            "onnxscript.function_libs.torch_lib.ops.fft": fft,
            "onnxscript.function_libs.torch_lib.ops.linalg": linalg,
            "onnxscript.function_libs.torch_lib.ops.nn": nn,
            "onnxscript.function_libs.torch_lib.ops.prims": prims,
            "onnxscript.function_libs.torch_lib.ops.special": special,
            "onnxscript.function_libs.torch_lib.ops.vision": vision,
        }
        self._submodule = subs
        return subs

    def fallback(
        self,
        name: Any,
        fct: Optional[Callable],
        args: List[Any],
        kwargs: Dict[str, Any],
        builder: "GraphBuilder",  # noqa: F821
    ) -> Optional[Callable]:
        """
        The function is called after the function converting an aten function
        into ONNX. *fct* is this function. It can be changed and just
        set when mapping was found.

        :param name: object or str
        :param fct: function found so far
        :param args: known arguments coming from the graph module
        :param kwargs: known named arguments coming from the graph module
        :param builder: GraphBuilder
        :return: callable
        """
        if fct is not None:
            # The conversion has been found.
            return fct

        from onnxscript.function_libs.torch_lib.registration import default_registry

        if hasattr(name, "__qualname__") and "::" in name.__qualname__:
            key = name.__qualname__
        else:
            key = str(name)
        if key.startswith("aten."):
            key = "aten::" + key[6:]

        if key not in default_registry:
            if self.verbose > 2:
                print(
                    "[OxsDispatcher.fallback] unable to find any fallback for {name!r} or {key!r}"
                )
            return None

        regfct = default_registry[key]
        assert len(regfct.overloads) > 0, (
            f"Unable to find onnxscript submodule {fct.function.__module__!r}. "
            f"onnxscript has a function with no overloaded instances, "
            f"key={key!r}, name={name!r}{builder.get_debug_msg()}"
        )
        fct = regfct.overloads[0]
        assert fct.function.__module__ in self.submodules, (
            f"Unable to find onnxscript submodule {fct.function.__module__!r}. "
            f"The fallback to onnxscript is not implemented yet for function "
            f"key={key!r}, name={name!r}{builder.get_debug_msg()}"
        )

        if self.verbose > 3:
            print(
                f"[OxsDispatcher.fallback] found {len(regfct.overloads)} "
                f"overloads for {key!r} ({name!r}), taking the first one."
            )

        def wrapper(g, sts, outputs, *args, _fct=fct, _dispatcher=self, **kwargs):
            op = OxsOpset(g)
            vargs = [(Var(x) if isinstance(x, str) else x) for x in args]

            # rewrite op, Rank, IsScalar in every submodule
            old = self._update_oxs(op)

            # call the function
            res = _fct.function(*vargs, **kwargs)

            # restore op, Rank, IsScalar
            self._restore_oxs(old)

            if isinstance(res, tuple):
                tres = tuple(r.name for r in res)
                cres = tres
            else:
                tres = res.name
                cres = (res.name,)

            if outputs is None:
                return tres

            # We need to rename.
            assert len(outputs) == len(cres), (
                f"Mismatched number of outputs, expecting {outputs!r} but got "
                f"{len(cres)} from {name!r} (key={key!r}){g.get_debug_msg()}"
            )

            for r, o in zip(cres, outputs):
                builder.op.Identity(r, outputs=[o], name=key.replace("::", "."))

            if len(outputs) == 1:
                return outputs[0]
            return tuple(outputs)

        return wrapper

    def _update_oxs(self, op: OxsOpset):
        keep = {}
        for k, v in self.submodules.items():
            old = v.op, getattr(v, "IsScalar", None), getattr(v, "Rank", None)
            v.op = op
            v.Rank = op.Rank
            v.IsScalar = op.IsScalar
            keep[k] = old
        return keep

    def _restore_oxs(self, old):
        for k, v in self.submodules.items():
            v.op, v.IsScalar, v.Rank = old[k]


class OxsDebugDispatcher(OxsDispatcher):
    """
    Tries the fallback even if is not necessary to check
    it is working.

    :param verbose: verbosity
    :param raise_exc: fail or raise an exception

    The class can be used the following way.

    .. runpython::
        :showcode:
        :process:

        import torch
        from experimental_experiment.torch_helper.llama_helper import get_llama_model
        from experimental_experiment.xbuilder import OptimizationOptions
        from experimental_experiment.torch_interpreter import to_onnx
        from experimental_experiment.torch_interpreter.oxs_dispatcher import (
            OxsDebugDispatcher,
        )

        with torch.no_grad():
            model, input_tensors = get_llama_model()
            input_tensors = input_tensors[0]

            to_onnx(
                model,
                input_tensors,
                input_names=[f"input{i}" for i in range(len(input_tensors))],
                options=OptimizationOptions(patterns=None),
                verbose=0,
                dispatcher=OxsDebugDispatcher(verbose=2, raise_exc=False),
            )"""

    def __init__(self, verbose: int = 0, raise_exc: bool = True):
        super(OxsDispatcher, self).__init__({}, verbose=verbose)
        self._submodule = None
        self.raise_exc = raise_exc

    def fallback(
        self,
        name: Any,
        fct: Optional[Callable],
        args: List[Any],
        kwargs: Dict[str, Any],
        builder: "GraphBuilder",  # noqa: F821
    ) -> Optional[Callable]:
        if self.raise_exc:
            res = OxsDispatcher.fallback(self, name, None, args, kwargs, builder)
            res(builder, False, None, *args, **kwargs)
            if self.verbose > 1:
                print(
                    f"[OxsDebugDispatcher.fallback] fallback "
                    f"verified for {name!r}: {res}"
                )
        else:
            try:
                res = OxsDispatcher.fallback(self, name, None, args, kwargs, builder)
            except (
                AssertionError,
                AttributeError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as e:
                if self.verbose > 1:
                    print(
                        f"[OxsDebugDispatcher.fallback] fallback "
                        f"failed for {name!r} with e={e}"
                    )
                return fct
            try:
                res(builder, False, None, *args, **kwargs)
            except (
                AssertionError,
                AttributeError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as e:
                if self.verbose > 1:
                    print(
                        f"[OxsDebugDispatcher.fallback] fallback "
                        f"unverified for {name!r} with e={e}"
                    )
                return fct
        if self.verbose > 1:
            print(
                f"[OxsDebugDispatcher.fallback] fallback verified "
                f"for {name!r} with {res}"
            )
        return fct or res
