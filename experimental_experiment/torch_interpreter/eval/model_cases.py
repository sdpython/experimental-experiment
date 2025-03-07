import packaging.version as pv
import numpy as np
import torch


class AtenRollRelu(torch.nn.Module):
    def forward(self, x):
        return torch.relu(torch.roll(x, -1, -1))

    _inputs = ((torch.arange(8 * 3) + 10).reshape((2, -1, 4)).to(torch.float32),)
    _dynamic = {"x": {0: torch.export.Dim("batch")}}


class AtenRollPos(torch.nn.Module):
    def forward(self, x):
        return torch.roll(x, 1, -1)

    _inputs = ((torch.arange(8 * 3) + 10).reshape((2, -1, 4)).to(torch.float32),)
    _dynamic = {"x": {0: torch.export.Dim("batch")}}


class InplaceAdd(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.bias = torch.ones((1, 4), dtype=torch.float32)

    def forward(self, x):
        x += self.bias
        return x

    _inputs = [(torch.rand(3, 4),), (torch.rand(5, 4),)]
    _dynamic = {"x": {0: torch.export.Dim("batch")}}


class InplaceAdd2(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.bias = torch.ones((1, 4), dtype=torch.float32)

    def forward(self, x):
        x.add_(self.bias)
        return x

    _inputs = [(torch.rand(3, 4),), (torch.rand(5, 4),)]
    _dynamic = {"x": {0: torch.export.Dim("batch")}}


class InplaceAdd_Mul(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.bias = torch.ones((1, 4), dtype=torch.float32)

    def forward(self, x):
        x.add_(self.bias)
        return x * 2

    _inputs = [(torch.rand(3, 4),), (torch.rand(5, 4),)]
    _dynamic = {"x": {0: torch.export.Dim("batch")}}


class InplaceCloneAdd_(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.bias = torch.ones((1, 4), dtype=torch.float32)

    def forward(self, x):
        x = x.clone()
        x.add_(self.bias)
        return x

    _inputs = [(torch.rand(3, 4),), (torch.rand(5, 4),)]
    _dynamic = {"x": {0: torch.export.Dim("batch")}}


class InplaceSetItemSquare(torch.nn.Module):

    def forward(self, x):
        x[:2, :3] = 1
        return x

    _inputs = [(torch.rand(5, 5),), (torch.rand(7, 5),)]
    _dynamic = {"x": {0: torch.export.Dim("batch")}}


class InplaceSetItemSquareAdd(torch.nn.Module):

    def forward(self, x):
        x[:2, :3] = 1
        return x + 2

    _inputs = [(torch.rand(5, 5),), (torch.rand(7, 5),)]
    _dynamic = {"x": {0: torch.export.Dim("batch")}}


class InplaceSetItemSquareAdd2(torch.nn.Module):

    def forward(self, x):
        x[:2, :3] = 1
        return x + 2, x + 3

    _inputs = [(torch.rand(5, 5),), (torch.rand(7, 5),)]
    _dynamic = {"x": {0: torch.export.Dim("batch")}}


class InplaceSetItemEllipsis_1(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.params = torch.zeros((1, 8192, 4), dtype=torch.float32)

    def forward(self, index, update):
        copy = self.params.clone()
        copy[..., index] = update
        return copy

    _inputs = (
        (torch.from_numpy(np.array([0, 3, 2, 1])).to(torch.int64)),
        (torch.arange(4 * 8192) + 10).reshape((-1, 4)).to(torch.float32),
    )
    _dynamic = {"x": {0: torch.export.Dim("batch")}}


class InplaceSetItemEllipsis_2(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.params = torch.zeros((1, 8192, 6), dtype=torch.float32)

    def forward(self, index, update):
        copy = self.params.clone()
        copy[..., index] = update
        return copy

    _inputs = (
        torch.from_numpy(np.array([0, 3, 2, 5])).to(torch.int64),
        (torch.arange(4 * 8192) + 10).reshape((-1, 4)).to(torch.float32),
    )
    _dynamic = {"x": {0: torch.export.Dim("batch")}}


class InplaceSetItemMask(torch.nn.Module):
    def forward(self, x):
        mask = x.to(bool)
        x[mask] = 2
        return x

    _inputs = [(torch.randn((2, 3, 3)),), (torch.randn((3, 3, 3)),)]
    _dynamic = {"x": {0: torch.export.Dim("batch")}}


class AtenInterpolate(torch.nn.Module):

    def forward(self, x):
        y = torch.nn.functional.interpolate(
            x,
            scale_factor=2.0,
            mode="bilinear",
            recompute_scale_factor=False,
        )
        return y

    _inputs = (torch.randn(2, 2, 3, 4, requires_grad=False),)
    _dynamic = {"x": {0: torch.export.Dim("batch")}}


class AtenNonZero(torch.nn.Module):

    def forward(self, x):
        y = torch.nonzero(x)
        return y

    _inputs = (torch.randn(3, 4, requires_grad=False),)
    _dynamic = {"x": {0: torch.export.Dim("batch")}}


class AtenNonZeroTuple(torch.nn.Module):

    def forward(self, x):
        y = torch.nonzero(x, as_tuple=True)
        return y[0], y[1]

    _inputs = (torch.randn(3, 4, requires_grad=False),)
    _dynamic = {"x": {0: torch.export.Dim("batch")}}


class AtenAsStrided(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = torch.as_strided(x, (2, 2, 8, 4), (128, 8, 16, 1))
        return y

    _inputs = (torch.randn((2, 2, 8, 8), requires_grad=False),)
    _dynamic = {"x": {0: torch.export.Dim("batch")}}


class ComplexPolar(torch.nn.Module):
    def forward(self, x, angle):
        return torch.polar(x, angle)

    _inputs = (torch.rand(4, 4), torch.rand(4, 4))
    _dynamic = {"x": {0: torch.export.Dim("batch")}, "angle": {0: torch.export.Dim("batch")}}


class ControlFlowCond(torch.nn.Module):
    def forward(self, x):
        def true_fn(x):
            return torch.sin(x)

        def false_fn(x):
            return torch.cos(x)

        return torch.cond(x.sum() > 0, true_fn, false_fn, [x])

    _inputs = (torch.rand(5, 3),)
    _dynamic = {"x": {0: torch.export.Dim("batch")}}


class ControlFlowCond2Outputs(torch.nn.Module):
    def forward(self, x):
        def true_fn(x):
            return torch.sin(x), torch.cos(x)

        def false_fn(x):
            return torch.cos(x), torch.sin(x)

        return torch.cond(x.sum() > 0, true_fn, false_fn, [x])

    _inputs = (torch.rand(5, 3),)
    _dynamic = {"x": {0: torch.export.Dim("batch")}}


class ControlFlowCond2Inputs(torch.nn.Module):
    def forward(self, x, y):
        def true_fn(x, y):
            return torch.sin(x), torch.cos(x) + y

        def false_fn(x, y):
            return torch.cos(x), torch.sin(x) + y

        return torch.cond(x.sum() > 0, true_fn, false_fn, [x, y])

    _inputs = torch.rand(5, 3), torch.rand(5, 3)
    _dynamic = {"x": {0: torch.export.Dim("batch")}, "y": {0: torch.export.Dim("batch")}}


class ControlFlowNestCond(torch.nn.Module):
    def forward(self, x):
        def true_fn2(x):
            def true_fn1(x):
                return torch.sin(x)

            def false_fn1(x):
                return torch.cos(x)

            return torch.cond(x.sum() < 0, true_fn1, false_fn1, [x])

        def false_fn2(x):
            return -x

        return torch.cond(x.sum() > 0, true_fn2, false_fn2, [x])

    _inputs = (torch.rand(5, 3),)
    _dynamic = {"x": {0: torch.export.Dim("batch")}}


class ControlFlowCondConstant(torch.nn.Module):
    def forward(self, x):
        def true_fn(x):
            return torch.sin(x) - torch.ones(x.shape, dtype=x.dtype)

        def false_fn(x):
            return torch.cos(x) + torch.ones((1, 1024), dtype=x.dtype)

        return torch.cond(x.sum() > 0, true_fn, false_fn, [x])

    _inputs = (torch.rand(1024, 1024),)
    _dynamic = {"x": {0: torch.export.Dim("batch")}}


class ControlFlowCondNestedModule(torch.nn.Module):

    class Submodule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Nested weight
            self.weight = torch.nn.Parameter(torch.tensor([100.0]))

        def forward(self, x):
            def true_fn(x):
                return x * self.weight

            def false_fn(x):
                return x / self.weight

            y = torch.cond(torch.abs(x).sum() > 100, true_fn, false_fn, [x])
            return y

    def __init__(self):
        super().__init__()
        self.submodule = ControlFlowCondNestedModule.Submodule()
        self.weight = torch.nn.Parameter(torch.tensor([42.0]))

    def forward(self, x):
        def true_fn(x):
            return self.submodule(x)

        def false_fn(x):
            return x - self.weight

        y = torch.cond(x.sum() > 0, true_fn, false_fn, [x])
        return y

    _inputs = (torch.tensor([-1, 2]),)
    _dynamic = {"x": {0: torch.export.Dim("batch")}}


class ControlFlowCondNonZero(torch.nn.Module):
    def forward(self, input_ids, image_features, vocab_size):
        def then_branch(input_ids, image_features, vocab_size):
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])

            condition = (input_ids < 0) & (input_ids > -int(1e9))
            positions = torch.nonzero(condition, as_tuple=True)
            input_ids = input_ids.clamp_min(0).clamp_max(vocab_size)
            return (input_ids, positions[0], positions[1])

        def else_branch(input_ids, image_features, vocab_size):
            r = torch.where(torch.zeros((1, 1), dtype=torch.bool))
            return (input_ids, r[0], r[1])

        a, b, c = torch.cond(
            image_features.numel() > 0,
            then_branch,
            else_branch,
            [input_ids, image_features, vocab_size],
        )
        return a, b, c

    _inputs = [
        (
            (torch.arange(24) - 8).reshape((2, -1)).to(torch.int64),
            torch.arange(32).reshape((2, -1)).to(torch.float32),
            1025,
        ),
        (
            (torch.arange(24) - 8).reshape((2, -1)).to(torch.int64),
            torch.tensor([[], []], dtype=torch.float32),
            1025,
        ),
    ]
    _dynamic = (
        {0: torch.export.Dim("batch")},
        {0: torch.export.Dim("batch"), 1: torch.export.Dim("seq_length")},
        None,
    )


if pv.Version(".".join(torch.__version__.split(".")[:2])) >= pv.Version("2.7"):

    class ControlFlowScan(torch.nn.Module):

        @staticmethod
        def add(carry: torch.Tensor, y: torch.Tensor):
            next_carry = carry + y
            return [next_carry, next_carry]

        def forward(self, x):
            init = torch.zeros_like(x[0])
            carry, out = torch.ops.higher_order.scan(
                ControlFlowScan.add, [init], [x], additional_inputs=[]
            )
            return carry

        _inputs = (torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32),)
        _dynamic = {"x": {0: torch.export.Dim("batch")}}

    class ControlFlowScan2Carried(torch.nn.Module):
        @staticmethod
        def add(
            carry1: torch.Tensor, carry2: torch.Tensor, y1: torch.Tensor, y2: torch.Tensor
        ):
            next_carry1 = carry1 + y1
            next_carry2 = carry2 * y2
            return [next_carry1, next_carry2, next_carry1, next_carry2]

        def forward(self, x):
            init1 = torch.zeros_like(x[0])
            init2 = torch.ones_like(x[0])
            carry1, carry2, out1, out2 = torch.ops.higher_order.scan(
                ControlFlowScan2Carried.add,
                [init1, init2],
                [x, x * 2],
                # dim=0,  # 01/31/2025, not supported anymore
                additional_inputs=[],
            )
            return carry1, carry2, out1, out2

        _inputs = (
            torch.tensor([[1, 2, 3, -1], [4, 5, 6, -1], [7, 8, 9, -1]], dtype=torch.float32),
        )
        _dynamic = {"x": {0: torch.export.Dim("batch")}}

    class ControlFlowScanCDist(torch.nn.Module):
        @staticmethod
        def dist(carry: torch.Tensor, x: torch.Tensor):
            sub = carry - x.reshape((1, -1))
            sq = sub * sub
            rd = sq.sum(axis=1) ** 0.5
            # clone --> UnsupportedAliasMutationException:
            # Combine_fn might be aliasing the input!
            return [carry.clone(), rd]

        def forward(self, x):
            carry, out = torch.ops.higher_order.scan(
                ControlFlowScanCDist.dist,
                [x],
                [x],
                # dim=0,  # 01/31/2025, not supported anymore
                additional_inputs=[],
            )
            return out

        _inputs = (
            torch.tensor([[1, 2, 3, -1], [4, 5, 6, -1], [7, 8, 9, -1]], dtype=torch.float32),
        )
        _dynamic = {"x": {0: torch.export.Dim("batch")}}

    class ControlFlowScanCDist2(torch.nn.Module):
        @staticmethod
        def dist(unused: torch.Tensor, x: torch.Tensor, samex: torch.Tensor):
            sub = samex - x.reshape((1, -1))
            sq = sub * sub
            rd = torch.sqrt(sq.sum(axis=1))
            # clone --> UnsupportedAliasMutationException:
            # Combine_fn might be aliasing the input!
            return [unused.clone(), rd]

        def forward(self, x):
            z = torch.tensor([0], dtype=torch.float32)
            y = x.clone()
            out = torch.ops.higher_order.scan(
                ControlFlowScanCDist2.dist,
                [z],
                [x],
                # dim=0,  # 01/31/2025, not supported anymore
                additional_inputs=[y],
            )
            return out[1]

        _inputs = (
            torch.tensor([[1, 2, 3, -1], [4, 5, 6, -1], [7, 8, 9, -1]], dtype=torch.float32),
        )
        _dynamic = {"x": {0: torch.export.Dim("batch")}}

    class ControlFlowScanCDistXY(torch.nn.Module):

        @staticmethod
        def dist(y: torch.Tensor, scanned_x: torch.Tensor):
            sub = y - scanned_x.reshape((1, -1))
            sq = sub * sub
            rd = torch.sqrt(sq.sum(axis=1))
            # clone --> UnsupportedAliasMutationException:
            # Combine_fn might be aliasing the input!
            return [y.clone(), rd]

        def forward(self, x, y):
            carry, out = torch.ops.higher_order.scan(
                ControlFlowScanCDistXY.dist,
                [y],
                [x],
                # dim=0,  # 01/31/2025, not supported anymore
                additional_inputs=[],
            )
            return out

        _inputs = [
            (torch.randn(3, 4), torch.randn(5, 4)),
            (torch.randn(13, 14), torch.randn(15, 14)),
        ]
        _dynamic = {
            "x": {0: torch.export.Dim("x_rows"), 1: torch.export.Dim("dim")},
            "y": {0: torch.export.Dim("y_rows"), 1: torch.export.Dim("dim")},
        }

else:

    class ControlFlowScan(torch.nn.Module):

        @staticmethod
        def add(carry: torch.Tensor, y: torch.Tensor):
            next_carry = carry + y
            return [next_carry, next_carry]

        def forward(self, x):
            init = torch.zeros_like(x[0])
            carry, out = torch.ops.higher_order.scan(
                ControlFlowScan.add, [init], [x], dim=0, additional_inputs=[]
            )
            return carry

        _inputs = (torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32),)
        _dynamic = {"x": {0: torch.export.Dim("batch")}}

    class ControlFlowScan2Carried(torch.nn.Module):
        @staticmethod
        def add(
            carry1: torch.Tensor, carry2: torch.Tensor, y1: torch.Tensor, y2: torch.Tensor
        ):
            next_carry1 = carry1 + y1
            next_carry2 = carry2 * y2
            return [next_carry1, next_carry2, next_carry1, next_carry2]

        def forward(self, x):
            init1 = torch.zeros_like(x[0])
            init2 = torch.ones_like(x[0])
            carry1, carry2, out1, out2 = torch.ops.higher_order.scan(
                ControlFlowScan2Carried.add,
                [init1, init2],
                [x, x * 2],
                dim=0,
                additional_inputs=[],
            )
            return carry1, carry2, out1, out2

        _inputs = (
            torch.tensor([[1, 2, 3, -1], [4, 5, 6, -1], [7, 8, 9, -1]], dtype=torch.float32),
        )
        _dynamic = {"x": {0: torch.export.Dim("batch")}}

    class ControlFlowScanCDist(torch.nn.Module):
        @staticmethod
        def dist(carry: torch.Tensor, x: torch.Tensor):
            sub = carry - x.reshape((1, -1))
            sq = sub * sub
            rd = sq.sum(axis=1) ** 0.5
            # clone --> UnsupportedAliasMutationException:
            # Combine_fn might be aliasing the input!
            return [carry.clone(), rd]

        def forward(self, x):
            carry, out = torch.ops.higher_order.scan(
                ControlFlowScanCDist.dist,
                [x],
                [x],
                dim=0,
                additional_inputs=[],
            )
            return out

        _inputs = (
            torch.tensor([[1, 2, 3, -1], [4, 5, 6, -1], [7, 8, 9, -1]], dtype=torch.float32),
        )
        _dynamic = {"x": {0: torch.export.Dim("batch")}}

    class ControlFlowScanCDist2(torch.nn.Module):
        @staticmethod
        def dist(unused: torch.Tensor, x: torch.Tensor, samex: torch.Tensor):
            sub = samex - x.reshape((1, -1))
            sq = sub * sub
            rd = torch.sqrt(sq.sum(axis=1))
            # clone --> UnsupportedAliasMutationException:
            # Combine_fn might be aliasing the input!
            return [unused.clone(), rd]

        def forward(self, x):
            z = torch.tensor([0], dtype=torch.float32)
            y = x.clone()
            out = torch.ops.higher_order.scan(
                ControlFlowScanCDist2.dist,
                [z],
                [x],
                dim=0,
                additional_inputs=[y],
            )
            return out[1]

        _inputs = (
            torch.tensor([[1, 2, 3, -1], [4, 5, 6, -1], [7, 8, 9, -1]], dtype=torch.float32),
        )
        _dynamic = {"x": {0: torch.export.Dim("batch")}}

    class ControlFlowScanCDistXY(torch.nn.Module):

        @staticmethod
        def dist(y: torch.Tensor, scanned_x: torch.Tensor):
            sub = y - scanned_x.reshape((1, -1))
            sq = sub * sub
            rd = torch.sqrt(sq.sum(axis=1))
            # clone --> UnsupportedAliasMutationException:
            # Combine_fn might be aliasing the input!
            return [y.clone(), rd]

        def forward(self, x, y):
            carry, out = torch.ops.higher_order.scan(
                ControlFlowScanCDistXY.dist,
                [y],
                [x],
                dim=0,
                additional_inputs=[],
            )
            return out

        _inputs = [
            (torch.randn(3, 4), torch.randn(5, 4)),
            (torch.randn(13, 14), torch.randn(15, 14)),
        ]
        _dynamic = {
            "x": {0: torch.export.Dim("x_rows"), 1: torch.export.Dim("dim")},
            "y": {0: torch.export.Dim("y_rows"), 1: torch.export.Dim("dim")},
        }


class SignatureInt1(torch.nn.Module):
    def __init__(self, n_dims: int = 3, n_targets: int = 1):
        super().__init__()
        self.linear = torch.nn.Linear(n_dims, n_targets)
        self.buff = torch.nn.parameter.Buffer(torch.tensor([0.5] * n_targets))

    def forward(self, x, i: int = 2):
        return torch.sigmoid(self.linear(x)) - self.buff + x[:, i : i + 1]

    _inputs = [
        ((torch.arange(4 * 3) + 10).reshape((-1, 3)).to(torch.float32), 1),
        ((torch.arange(8 * 3) + 10).reshape((-1, 3)).to(torch.float32), 2),
    ]
    _dynamic = ({0: torch.export.Dim("batch", min=1, max=1024)}, None)


class SignatureFloat1(torch.nn.Module):
    def __init__(self, n_dims: int = 3, n_targets: int = 1):
        super().__init__()
        self.linear = torch.nn.Linear(n_dims, n_targets)
        self.buff = torch.nn.parameter.Buffer(torch.tensor([0.5] * n_targets))

    def forward(self, x, alpha: float = 2.0):
        return torch.sigmoid(self.linear(x)) - self.buff * alpha

    _inputs = [
        ((torch.arange(4 * 3) + 10).reshape((-1, 3)).to(torch.float32), 1.5),
        ((torch.arange(8 * 3) + 10).reshape((-1, 3)).to(torch.float32), 2.5),
    ]
    _dynamic = ({0: torch.export.Dim("batch", min=1, max=1024)}, None)


class SignatureInt2(torch.nn.Module):
    def __init__(self, n_dims: int = 3, n_targets: int = 1):
        super().__init__()
        self.linear = torch.nn.Linear(n_dims, n_targets)
        self.buff = torch.nn.parameter.Buffer(torch.tensor([0.5] * n_targets))

    def forward(self, x, i: int = 2):
        return torch.sigmoid(self.linear(x)) - self.buff + x[:, i]

    _inputs = ((torch.arange(4 * 3) + 10).reshape((-1, 3)).to(torch.float32), 1)
    _dynamic = {
        "x": {0: torch.export.Dim("batch")},
        "i": None,  # torch.export.Dim("ii", min=0, max=3)}
    }


class SignatureListFixedLength(torch.nn.Module):
    def __init__(self, n_dims: int = 3, n_targets: int = 1):
        super().__init__()
        self.linear = torch.nn.Linear(n_dims, n_targets)
        self.buff = torch.nn.parameter.Buffer(torch.tensor([0.5] * n_targets))

    def forward(self, x, lx: list):
        return (
            torch.sigmoid(self.linear(x)) - self.buff + lx[0] * lx[1].sum(axis=1, keepdim=True)
        )

    _inputs = [
        (
            (torch.arange(4 * 3) + 10).reshape((-1, 3)).to(torch.float32),
            [
                (torch.arange(4) + 10).reshape((-1, 1)).to(torch.float32),
                (torch.arange(4 * 2) + 10).reshape((-1, 2)).to(torch.float32),
            ],
        ),
        (
            (torch.arange(8 * 3) + 10).reshape((-1, 3)).to(torch.float32),
            [
                (torch.arange(8) + 10).reshape((-1, 1)).to(torch.float32),
                (torch.arange(8 * 2) + 10).reshape((-1, 2)).to(torch.float32),
            ],
        ),
    ]
    _dynamic = {
        "x": {0: torch.export.Dim("batch")},
        "lx": [{0: torch.export.Dim("batch")}, {0: torch.export.Dim("batch")}],
    }


class SignatureListVariableLength(torch.nn.Module):
    def __init__(self, n_dims: int = 3, n_targets: int = 1):
        super().__init__()
        self.linear = torch.nn.Linear(n_dims, n_targets)
        self.buff = torch.nn.parameter.Buffer(torch.tensor([0.5] * n_targets))

    def forward(self, x, lx: list):
        t = torch.cat(lx, dim=1).sum(axis=1, keepdim=True)
        return torch.sigmoid(self.linear(x)) - self.buff + t

    _inputs = [
        (
            (torch.arange(4 * 3) + 10).reshape((-1, 3)).to(torch.float32),
            [
                (torch.arange(4) + 10).reshape((-1, 1)).to(torch.float32),
                (torch.arange(4 * 2) + 10).reshape((-1, 2)).to(torch.float32),
            ],
        ),
        (
            (torch.arange(8 * 3) + 10).reshape((-1, 3)).to(torch.float32),
            [
                (torch.arange(8) + 10).reshape((-1, 1)).to(torch.float32),
                (torch.arange(8 * 2) + 10).reshape((-1, 2)).to(torch.float32),
                (torch.arange(8 * 3) + 10).reshape((-1, 3)).to(torch.float32),
            ],
        ),
    ]
    _dynamic = {
        "x": {0: torch.export.Dim("batch")},
        "lx": [{0: torch.export.Dim("batch")}, {0: torch.export.Dim("batch")}],
    }


class BuildInLen(torch.nn.Module):
    def __init__(self, n_dims: int = 3, n_targets: int = 1):
        super().__init__()
        self.linear = torch.nn.Linear(n_dims, n_targets)
        self.buff = torch.nn.parameter.Buffer(torch.tensor([0.5] * n_targets))

    def forward(self, x, lx: list):
        t = lx[0] * lx[1].sum(axis=1, keepdim=True)
        if len(lx) > 2:
            t = t + lx[2].sum(axis=1, keepdim=True)
        return torch.sigmoid(self.linear(x)) - self.buff + t

    _inputs = [
        (
            (torch.arange(4 * 3) + 10).reshape((-1, 3)).to(torch.float32),
            [
                (torch.arange(4) + 10).reshape((-1, 1)).to(torch.float32),
                (torch.arange(4 * 2) + 10).reshape((-1, 2)).to(torch.float32),
            ],
        ),
        (
            (torch.arange(8 * 3) + 10).reshape((-1, 3)).to(torch.float32),
            [
                (torch.arange(8) + 10).reshape((-1, 1)).to(torch.float32),
                (torch.arange(8 * 2) + 10).reshape((-1, 2)).to(torch.float32),
                (torch.arange(8 * 3) + 10).reshape((-1, 3)).to(torch.float32),
            ],
        ),
    ]
    _dynamic = {
        "x": {0: torch.export.Dim("batch")},
        "lx": [{0: torch.export.Dim("batch")}, {0: torch.export.Dim("batch")}],
    }


class BuildInIsInstance(torch.nn.Module):
    def __init__(self, n_dims: int = 3, n_targets: int = 1):
        super().__init__()
        self.linear = torch.nn.Linear(n_dims, n_targets)
        self.buff = torch.nn.parameter.Buffer(torch.tensor([0.5] * n_targets))

    def forward(self, x, lx: list | torch.Tensor):
        if isinstance(lx, list):
            t = lx[0] * lx[1].sum(axis=1, keepdim=True)
            return torch.sigmoid(self.linear(x)) - self.buff + t
        return torch.sigmoid(self.linear(x)) - self.buff + lx

    _inputs = [
        (
            (torch.arange(4 * 3) + 10).reshape((-1, 3)).to(torch.float32),
            [
                (torch.arange(4) + 10).reshape((-1, 1)).to(torch.float32),
                (torch.arange(4 * 2) + 10).reshape((-1, 2)).to(torch.float32),
            ],
        ),
        (
            (torch.arange(8 * 3) + 10).reshape((-1, 3)).to(torch.float32),
            [
                (torch.arange(8) + 10).reshape((-1, 1)).to(torch.float32),
                (torch.arange(8 * 2) + 10).reshape((-1, 2)).to(torch.float32),
            ],
        ),
    ]
    _dynamic = {
        "x": {0: torch.export.Dim("batch")},
        "lx": [{0: torch.export.Dim("batch")}, {0: torch.export.Dim("batch")}],
    }


class SignatureShapeAsIndex(torch.nn.Module):
    def __init__(self, n_dims: int = 3, n_targets: int = 1):
        super().__init__()
        self.linear = torch.nn.Linear(n_dims, n_targets)
        self.buff = torch.nn.parameter.Buffer(torch.tensor([0.5] * n_targets))

    def forward(self, x, y):
        t = torch.sigmoid(self.linear(x)) + x
        return t[:, : y.shape[1]]

    _inputs = (
        (torch.arange(4 * 3) + 10).reshape((-1, 3)).to(torch.float32),
        (torch.arange(4 * 2) + 10).reshape((-1, 2)).to(torch.float32),
    )
    _dynamic = {
        "x": {0: torch.export.Dim("batch", min=0, max=1024)},
        "y": {
            0: torch.export.Dim("batch", min=0, max=1024),
            1: torch.export.Dim("length", min=0, max=2),
        },
    }


class TypeBFloat16(torch.nn.Module):

    def forward(self, x):
        xb = x.to(torch.bfloat16)
        return (xb + xb).to(torch.float32)

    _inputs = (torch.rand(4, 4).to(torch.float32),)
    _dynamic = {"x": {0: torch.export.Dim("batch")}}


class CropLastDimensionWithTensorShape(torch.nn.Module):

    def forward(self, x, y):
        return x[..., : y.shape[0]]

    _inputs = [
        (
            torch.rand(3, 4, 4).to(torch.float32),
            torch.rand(
                2,
            ).to(torch.float32),
        ),
        (
            torch.rand(6, 4, 4).to(torch.float32),
            torch.rand(
                3,
            ).to(torch.float32),
        ),
    ]
    _dynamic = {
        "x": {0: torch.export.Dim("batch")},
        "y": {0: torch.export.Dim("crop", min=1, max=3)},
    }


class CropLastDimensionWithTensorContent(torch.nn.Module):

    def forward(self, x, shape):
        return x[..., : shape[0]]

    _inputs = [
        (torch.rand(3, 4, 4).to(torch.float32), torch.tensor([2], dtype=torch.int64)),
        (torch.rand(6, 4, 4).to(torch.float32), torch.tensor([3], dtype=torch.int64)),
    ]
    _dynamic = {"x": {0: torch.export.Dim("batch")}}


class SignatureListFixedWithNone(torch.nn.Module):

    def forward(self, lx):
        x = lx[0]
        if lx[1] is not None:
            x += lx[1]
        if lx[2] is not None:
            x += lx[2]
        return x

    _inputs = [
        ([torch.rand((4, 4)), torch.rand((4, 4)), None],),
        ([torch.rand((4, 4)), torch.rand((4, 4)), torch.rand((4, 4))],),
    ]
    _dynamic = {
        "lx": [{0: torch.export.Dim("batch")}, {0: torch.export.Dim("batch")}],
    }


class CreateFromShape(torch.nn.Module):
    def forward(self, x):
        y = torch.ones((x.shape[0], x.shape[1] + 1))
        return y

    _inputs = [(torch.rand((4, 4)),), (torch.rand((5, 5)),)]
    _dynamic = {"x": {0: torch.export.Dim("dx"), 1: torch.export.Dim("dy")}}


class CreateFromShapeThroughFunction(torch.nn.Module):
    @staticmethod
    def add_one(dim: int) -> int:
        return dim + 1

    def forward(self, x):
        dy1 = CreateFromShapeThroughFunction.add_one(x.shape[1])
        y = torch.ones((x.shape[0], dy1))
        return y

    _inputs = [(torch.rand((4, 4)),), (torch.rand((5, 5)),)]
    _dynamic = {"x": {0: torch.export.Dim("dx"), 1: torch.export.Dim("dy")}}
