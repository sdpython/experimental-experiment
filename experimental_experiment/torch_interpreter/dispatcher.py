from typing import Any, Callable, Dict, List, Optional


class Dispatcher:
    """
    Used to changes the way class :class:`DynamoInterpreter
    <experimental_experiment.torch_interpreter.DynamoInterpreter>`
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
            if self.verbose > 2:
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
            if self.verbose > 2:
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
