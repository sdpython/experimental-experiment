import inspect
from typing import Any, Callable, Dict, List, Optional


class Dispatcher:
    """
    Used to changes the way class :class:`DynamoInterpreter
    <experimental_experiment.torch_interpreter.interpreter.DynamoInterpreter>`
    selects the function translating aten function or module.

    :param registered_functions: registered functions
    :param verbose: verbose
    """

    def __init__(self, registered_functions: Dict[str, Callable], verbose: int = 0):
        self.registered_functions = registered_functions
        self.verbose = verbose

    def _get_function_name(self, name: Any) -> str:
        if isinstance(name, str):
            return name

        if isinstance(name, type(abs)):
            new_name = f"aten_{name.__name__.replace('.', '_')}"
            if new_name in self.registered_functions:
                return new_name

        lookup_names = ["__qualname__", "__name__"]
        for att in lookup_names:
            if hasattr(name, att):
                v = getattr(name, att).replace(".", "_")
                if v in self.registered_functions:
                    return v

        return str(v)

    def find_function(self, name: Any) -> Optional[Callable]:
        """
        Finds the most suitable function to translate a function.

        :param name: function name or definition
        :return: the function or None if not found

        The signature of the returned function is similar to a function
        such as :func:`aten_elu
        <experimental_experiment.torch_interpreter._aten_functions.aten_elu>`.
        """
        key = self._get_function_name(name)
        if key not in self.registered_functions:
            if self.verbose > 3:
                print(
                    f"[Dispatcher.find_function] could not find a function for key={key!r} with name={name!r}"
                )
            return None

        return self.registered_functions[key]

    def find_method(self, name: Any) -> Optional[Callable]:
        """
        Finds the most suitable function to translate a method.

        :param name: method name or definition
        :return: the function or None if not found

        The signature of the returned function is similar to a function
        such as :func:`aten_elu
        <experimental_experiment.torch_interpreter._aten_functions.aten_elu>`.
        """
        if name not in self.registered_functions:
            if self.verbose > 3:
                print(
                    f"[Dispatcher.find_method] could not find a method for name={name!r}"
                )
            return None

        return self.registered_functions[name]

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
        return fct


class ForceDispatcher(Dispatcher):
    """
    Implements a dispatcher which as an onnx as it is
    when no converting function is found.

    :param signatures: function used only for their signature mapping
        a name to a function in order to have parameter names
    :param verbose: verbose
    :param domain: domain of the added node
    :param version: version of the domain
    :param strict: when an input is not a tensor, it becomes a named parameter
        if strict is False
    """

    def __init__(
        self,
        signatures: Optional[Dict[str, Callable]] = None,
        verbose: int = 0,
        domain: str = "aten.lib",
        version: int = 1,
        strict: bool = False,
    ):
        super(ForceDispatcher, self).__init__({}, verbose=verbose)
        self.signatures = signatures or {}
        self.domain = domain
        self.version = version
        self.strict = strict
        self._process_signatures()

    def _process_signature(self, f: Callable):
        args = []
        kwargs = []
        sig = inspect.signature(f)
        for name, p in sig.parameters.items():
            ann = p.annotation
            if p.default is inspect._empty:
                args.append(name)
            else:
                kwargs.append((name, p.default, None if ann is inspect._empty else ann))
        return args, kwargs

    def _process_signatures(self):
        self.sigs_ = {}
        for k, v in self.signatures.items():
            sig = self._process_signature(v)
            self.sigs_[k] = sig

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

        fname = self._get_function_name(name)

        def wrapper(
            g,
            sts,
            outputs,
            *args,
            _name=fname,
            _domain=self.domain,
            _version=self.version,
            **kwargs,
        ):
            sig = self.sigs_.get(_name, None)
            kwargs = kwargs.copy()
            new_args = []
            for i, n in enumerate(args):
                if isinstance(n, str):
                    new_args.append(n)
                    continue
                if isinstance(n, g.torch.Tensor):
                    init = g.make_initializer("", n)
                    new_args.append(init)
                    continue
                if not sig:
                    if self.strict:
                        raise RuntimeError(
                            f"Unsupported type {type(n)} for argument {i} for function {_name!r}{g.get_debug_msg()}"
                        )
                    kwargs[f"param_{i}"] = n
                    continue
                a, kw = sig
                assert i >= len(
                    a
                ), f"Unsupported type {type(n)} for argument {i} for function {_name!r}{g.get_debug_msg()}"
                ni = i - len(a)
                assert ni < len(
                    kw
                ), f"Unexpected argument at position {i}, for function {_name!r}{g.get_debug_msg()}"
                p = kw[ni]
                kwargs[p[0]] = n if p[2] is None else p[2](n)

            g.add_domain(_domain, _version)
            g.make_node(
                _name,
                new_args,
                outputs=outputs,
                domain=_domain,
                name=g.unique_node_name(_name),
                **kwargs,
            )
            if len(outputs) == 1:
                return outputs[0]
            return tuple(outputs)

        return wrapper
