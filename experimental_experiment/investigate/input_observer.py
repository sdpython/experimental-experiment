import contextlib
import inspect
import torch


class InputObserverInfo:
    def __init__(self):
        self.inputs_specs = []
        self.flat_inputs = []
        self.outputs_specs = []
        self.flat_outputs = []

    def __len__(self) -> int:
        return len(self.flat_inputs)

    def add_inputs(self, args, kwargs):
        flat_args, spec = torch.utils._pytree.tree_flatten((args, kwargs))
        self.inputs_specs.append(spec)
        self.flat_inputs.append([t.clone().detach() for t in flat_args])

    def add_outputs(self, res):
        flat_res, spec = torch.utils._pytree.tree_flatten(res)
        self.outputs_specs.append(spec)
        self.flat_outputs.append([t.clone().detach() for t in flat_res])


class InputObserver:
    def __init__(self, store_n_calls: int = 3):
        self.store_n_calls = store_n_calls
        self.info = None

    def _forward_captured(self, *args, _captured_forward=None, **kwargs):
        torch._check(_captured_forward is not None, lambda: "_captured_forward cannot be None")
        n_stored = len(self.info)
        if n_stored < self.store_n_calls:
            self.info.add_inputs(args, kwargs)
        res = _captured_forward(*args, **kwargs)
        if n_stored < self.store_n_calls:
            self.info.add_outputs(res)
        return res

    @contextlib.contextmanager
    def __call__(self, model: torch.nn.Module):
        if self.info is not None:
            raise RuntimeError(
                "This class was already used to capture a model. Please create a new one."
            )
        self.info = InputObserverInfo()
        self.info.signature = inspect.signature(model.forward)
        forward_method = model.forward
        model.forward = (
            lambda *args, _captured_forward=forward_method, **kwargs: self._forward_captured(
                *args, _captured_forward=_captured_forward, **kwargs
            )
        )
        try:
            yield self
        finally:
            model.forward = forward_method
