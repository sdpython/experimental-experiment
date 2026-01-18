import enum
import traceback
import torch

TRACING_TENSOR_INDEX = 0


class TracingTensorIndex:
    __slots__ = ["index"]

    def __init__(self, index: int | None = None):
        if index is None:
            global TRACING_TENSOR_INDEX
            TRACING_TENSOR_INDEX += 1
            self.index = TRACING_TENSOR_INDEX
        else:
            self.index = index


class TracingKind(enum.IntEnum):
    FUNCTION = 1
    OPS = 2


class TracingTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, data: torch.Tensor, **kwargs):
        global TRACING_TENSOR_INDEX
        assert not kwargs, f"Not implemented for kwargs={kwargs}"
        res = torch.Tensor._make_subclass(cls, data, data.requires_grad)
        res.__INDEX__ = TracingTensorIndex()
        return res

    @classmethod
    def wraps_with_tracing_tensor(cls, res):
        if res is None:
            return None
        if isinstance(res, tuple):
            return tuple(cls.wraps_with_tracing_tensor(t) for t in res)
        if type(res) is list:
            return [cls.wraps_with_tracing_tensor(t) for t in res]
        if type(res) is dict:
            return {k: cls.wraps_with_tracing_tensor(t) for k, t in res.items()}
        if type(res) is torch.Tensor:
            return TracingTensor(res)
        if type(res) is TracingTensor:
            return res
        if isinstance(res, (int, float, bool, str)):
            return res
        if type(res) in (torch.dtype, torch.device):
            return res
        raise NotImplementedError(
            f"wraps_with_tracing_tensor not implemented for type({type(res)})"
        )

    @classmethod
    def tracing_tensor_as_ids(cls, context, res):
        if res is None:
            return None
        if isinstance(res, tuple):
            return tuple(cls.tracing_tensor_as_ids(context, t) for t in res)
        if type(res) is list:
            return [cls.tracing_tensor_as_ids(context, t) for t in res]
        if type(res) is dict:
            return {k: cls.tracing_tensor_as_ids(context, t) for k, t in res.items()}
        if type(res) is TracingTensor:
            return res.__INDEX__
        if isinstance(res, torch.Tensor):
            context.store_tensor_constant(res)
            return res
        if isinstance(res, (int, float, bool, str)):
            return res
        if type(res) in (torch.dtype, torch.device):
            return res
        raise NotImplementedError(f"tracing_tensor_as_ids not implemented for type({type(res)})")


class TracingDispatch(torch.utils._python_dispatch.TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args, kwargs=None):
        stack = traceback.extract_stack()[:-1]
        self._enter_tracing_event(
            TracingKind.OPS, stack=stack, ops=func, types=types, args=args, kwargs=kwargs
        )
        res = func(*args, **(kwargs or {}))
        self._leave_tracing_event(TracingKind.OPS, ops=func, res=res)
        return TracingTensor.wraps_with_tracing_tensor(res)


class TracingFunction(torch.overrides.TorchFunctionMode):
    def __torch_function__(self, func, types, args, kwargs=None):
        stack = traceback.extract_stack()[:-1]
        self._enter_tracing_event(
            TracingKind.FUNCTION, stack=stack, ops=func, types=types, args=args, kwargs=kwargs
        )
        res = func(*args, **(kwargs or {}))
        self._leave_tracing_event(TracingKind.FUNCTION, ops=func, res=res)
        return TracingTensor.wraps_with_tracing_tensor(res)


class TracingContextEvent:
    def __init__(
        self,
        context,
        kind,
        stack=None,
        types=None,
        ops=None,
        args=(),
        kwargs=None,
        res=None,
    ):
        self.context = context
        self.level = context.level
        self.kind = kind
        self.stack = stack
        self.ops = ops
        self.args = TracingTensor.tracing_tensor_as_ids(context, args)
        self.kwargs = TracingTensor.tracing_tensor_as_ids(context, kwargs)
        self.res = TracingTensor.tracing_tensor_as_ids(context, res)
        self.types = types


class TracingContext:
    def __init__(self):
        self.tracing_dispatch = TracingDispatch()
        self.tracing_function = TracingFunction()

        self.tracing_dispatch._enter_tracing_event = self.enter_tracing_event
        self.tracing_dispatch._leave_tracing_event = self.leave_tracing_event

        self.tracing_function._enter_tracing_event = self.enter_tracing_event
        self.tracing_function._leave_tracing_event = self.leave_tracing_event

    def store_tensor_constant(self, tensor):
        self.constant_tensors.append(tensor)

    def __enter__(self):
        self.level = 0
        self.stack = []
        self.constant_tensors = []
        self.tracing_enter = self.tracing_dispatch.__enter__(), self.tracing_function.__enter__()
        return self.tracing_enter

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tracing_function.__exit__(exc_type, exc_val, exc_tb)
        self.tracing_dispatch.__exit__(exc_type, exc_val, exc_tb)
        return False

    def enter_tracing_event(self, kind, stack=None, ops=None, args=(), kwargs=None, types=None):
        self.level += 1
        self.stack.append(
            TracingContextEvent(
                self, kind, stack=stack, ops=ops, args=args, kwargs=kwargs, types=types
            )
        )
        return self

    def leave_tracing_event(self, kind, ops=None, res=None):
        self.stack.append(TracingContextEvent(self, kind, ops=ops, res=res))
        self.level -= 1
        return self
