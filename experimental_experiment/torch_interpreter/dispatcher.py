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
                    f"[Dispatcher.find_function] could not find a "
                    f"function for key={key!r} with name={name!r}"
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
                print(f"[Dispatcher.find_method] could not find a method for name={name!r}")
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
    :param only_registered: fails if a function is not found in signatures
    """

    def __init__(
        self,
        signatures: Optional[Dict[str, Callable]] = None,
        verbose: int = 0,
        domain: str = "aten.lib",
        version: int = 1,
        strict: bool = False,
        only_registered: bool = False,
    ):
        super().__init__({}, verbose=verbose)
        self.signatures = signatures or {}
        self.domain = domain
        self.version = version
        self.strict = strict
        self.only_registered = only_registered
        self._process_signatures()

    @classmethod
    def _convert_into_type(cls, annotation):
        assert (
            annotation is not None and annotation is not inspect._empty
        ), f"Unexpected annotation={annotation}"
        if annotation in (float, int, bool):
            return annotation
        if hasattr(annotation, "_name") and annotation._name == "List":
            assert len(annotation.__args__) == 1, f"Unexpected annotation {annotation}"
            assert annotation.__args__[0] in (float, int, bool), (
                f"Unexpected annotation {annotation}, "
                f"annotation.__args__[0]={annotation.__args__[0]!r}"
            )
            t = annotation.__args__[0]
            return lambda v, t=t: [t(_) for _ in v]

        raise RuntimeError(f"Unexpected annotation {annotation!r}")

    def _process_signature(self, f: Callable):
        args = []
        kwargs = []
        sig = inspect.signature(f)
        has_annotation = any(
            (p.annotation is not None and p.annotation is not inspect._empty)
            for p in sig.parameters.values()
        )
        # If there is annotation, we assume every result = None
        # without annotation is an optional Tensor.
        for name, p in sig.parameters.items():
            ann = p.annotation
            if p.default is inspect._empty:
                args.append(name)
            elif p.default is None:
                noann = p.annotation is None or p.annotation is inspect._empty
                if has_annotation and noann:
                    args.append(name)
                elif not noann:
                    kwargs.append(
                        (
                            name,
                            p.default,
                            (
                                None
                                if ann is inspect._empty or ann is None
                                else self._convert_into_type(ann)
                            ),
                        )
                    )
                else:
                    raise RuntimeError(
                        f"Unable to determine if parameter {name!r} "
                        f"is an input or a parameter, annotation is {p.annotation}, "
                        f"default is {p.default!r} for function {f}, "
                        f"has_annotation={has_annotation}"
                    )
            else:
                kwargs.append(
                    (
                        name,
                        p.default,
                        (
                            None
                            if ann is inspect._empty or ann is None
                            else self._convert_into_type(ann)
                        ),
                    )
                )
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
            _only_registered=self.only_registered,
            **kwargs,
        ):
            sig = self.sigs_.get(_name, None)
            assert (
                not _only_registered or sig is not None
            ), f"Unable to find a function with {_name!r}{g.get_debug_msg()}"
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
                            f"Unsupported type {type(n)} for argument {i} "
                            f"for function {_name!r}{g.get_debug_msg()}"
                        )
                    kwargs[f"param_{i}"] = n
                    continue

                a, kw = sig
                if n is None and i < len(a):
                    # An optional input.
                    new_args.append("")
                    continue

                assert i >= len(a), (
                    f"Unsupported type {type(n)} for argument {i} for function {_name!r}"
                    f"sig={sig}, {g.get_debug_msg()}"
                )
                ni = i - len(a)
                assert ni < len(kw), (
                    f"Unexpected argument at position {i}, for function {_name!r}"
                    f"sig={sig}{g.get_debug_msg()}"
                )
                p = kw[ni]
                kwargs[p[0]] = n if p[2] is None else p[2](n)

            # for some arguments given as named arguments
            new_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, g.torch.fx.node.Node):
                    new_args.append(v.name)
                    continue
                new_kwargs[k] = v

            # Let's get rid of the empty name at the end of the inputs.
            i = len(new_args) - 1
            while i >= 0 and new_args[i] == "":
                i -= 1
            new_args = new_args[: i + 1] if i >= 0 else []

            g.add_domain(_domain, _version)
            g.make_node(
                _name,
                new_args,
                outputs=outputs,
                domain=_domain,
                name=g.unique_node_name(_name),
                **new_kwargs,
            )
            if len(outputs) == 1:
                return outputs[0]
            return tuple(outputs)

        return wrapper
