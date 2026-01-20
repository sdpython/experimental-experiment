import contextlib
import inspect
import torch


def infer_dynamic_dimensions(shape_list: tuple[int, ...]) -> list[int]:
    """
    Returns the list dynamic dimensions given a list of shapes corresponding to the same tensor.
    """
    unique_ranks = {len(shape) for shape in shape_list}
    torch._check(
        len(unique_ranks) == 1, lambda: "all shapes in shape_list must have the the same rank"
    )
    rank = unique_ranks.pop()
    dynamic = []
    for i in range(rank):
        dims = [shape[i] for shape in shape_list]
        if len(set(dims)) > 1:
            dynamic.append(i)
    return dynamic


class InputObserverInfo:
    def __init__(self, signature):
        self.inputs_specs = []
        self.flat_inputs = []
        self.n_args = []
        self.kwargs_keys = []
        self.outputs_specs = []
        self.flat_outputs = []
        self.signature = signature

        self._max_args = None
        self._max_kwargs = None
        self._spec = None

    def __len__(self) -> int:
        return len(self.flat_inputs)

    def add_inputs(self, args, kwargs):
        kwargs = {
            k: v
            for k, v in kwargs.items()
            if v is not None and not isinstance(v, (int, float, bool))
        }
        flat_args, spec = torch.utils._pytree.tree_flatten((args, kwargs))
        self.inputs_specs.append(spec)
        self.n_args.append(len(args))
        self.kwargs_keys.append(tuple(kwargs))
        cloned = [(None if t is None else t.clone().detach()) for t in flat_args]
        self.flat_inputs.append(cloned)

        cloned_args, cloned_kwargs = torch.utils._pytree.tree_unflatten(cloned, spec)
        if self._max_args is None or len(cloned_args) > len(self._max_args):
            self._max_args = cloned_args
        if self._max_kwargs is None or len(cloned_kwargs) > len(self._max_kwargs):
            self._max_kwargs = cloned_kwargs

    def add_outputs(self, res):
        flat_res, spec = torch.utils._pytree.tree_flatten(res)
        self.outputs_specs.append(spec)
        self.flat_outputs.append([t.clone().detach() for t in flat_res])

    def build_inputs_completed_with_none_values(self):
        # Let's compute the sizes of each indenpendently.
        max_spec = torch.utils._pytree.tree_flatten((self._max_args, self._max_kwargs))[1]
        arg_sizes = [len(torch.utils._pytree.tree_flatten(a)[0]) for a in self._max_args]
        kwarg_sizes = {
            k: len(torch.utils._pytree.tree_flatten(v)[0]) for k, v in self._max_kwargs.items()
        }

        # Let's reprocess everything.
        new_flat_inputs = []
        for args_kwargs, spec in zip(self.flat_inputs, self.inputs_specs):
            args, kwargs = torch.utils._pytree.tree_unflatten(args_kwargs, spec)
            if len(set(kwargs) | set(self._max_kwargs)) > len(self._max_kwargs):
                raise RuntimeError(
                    "At least one call to the observed model "
                    "must contain all the named arguments."
                )
            flat = []
            for i in range(len(self._max_args)):
                if i < len(args):
                    flat.extend(torch.utils._pytree.tree_flatten(args[i])[0])
                else:
                    flat.extend([None for _ in range(arg_sizes[i])])
            for k in self._max_kwargs:
                if k in kwargs:
                    flat.extend(torch.utils._pytree.tree_flatten(kwargs[k])[0])
                else:
                    flat.extend([None for _ in range(kwarg_sizes[k])])
            new_flat_inputs.append(flat)
        return new_flat_inputs, max_spec

    def infer_dynamic_shapes(self):
        if not self.flat_inputs:
            raise RuntimeError("No inputs were captured.")
        flat_inputs, max_spec = self.build_inputs_completed_with_none_values()
        if len({len(flat) for flat in flat_inputs}) != 1:
            raise NotImplementedError(
                "infer_dynamic_shapes is not implemented "
                "when the number of input tensors are not the same."
            )
        shape_lists = [
            [(None if t is None else t.shape) for t in tensors] for tensors in flat_inputs
        ]
        n_tensors = len(shape_lists[0])
        dynamic_shapes = [
            infer_dynamic_dimensions(
                [s for s in [shapes[index] for shapes in shape_lists] if s is not None]
            )
            for index in range(n_tensors)
        ]
        cst = torch.export.Dim.DYNAMIC
        flat_dynamic_shapes = [dict.fromkeys(dims, cst) for dims in dynamic_shapes]
        if len(flat_dynamic_shapes) == len(self._max_args) + len(self._max_kwargs):
            # It means forward method is called with tensors only.
            if not self._max_kwargs:
                # only positional arguments
                return tuple(flat_dynamic_shapes)
            if not self._max_args:
                # only named arguments
                return dict(zip(list(self._max_kwargs), flat_dynamic_shapes))
            # positional arguments needs to be moved to the named arguments
            n_args = len(self._max_args)
            pos_names = list(self.signature.parameters)[:n_args]
            return {
                **dict(zip(pos_names, flat_dynamic_shapes[:n_args])),
                **dict(zip(list(self._max_kwargs), flat_dynamic_shapes[n_args:])),
            }
        # nested types, here comes the fun part because the the shapes cannot be unflattened,
        # custom classes must appear in their flattened shape.
        raise NotImplementedError(
            f"There are nested types in the inputs, "
            f"len(flat_dynamic_shapes)={len(flat_dynamic_shapes)}, "
            f"len(self._max_args)={len(self._max_args)}, "
            f"len(self._max_kwargs)={len(self._max_kwargs)}, "
            f"max_spec={max_spec}"
        )


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
        self.info = InputObserverInfo(signature=inspect.signature(model.forward))
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

    def infer_dynamic_shapes(self):
        return self.info.infer_dynamic_shapes()
