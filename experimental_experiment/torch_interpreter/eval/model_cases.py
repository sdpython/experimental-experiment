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

    _inputs = ((torch.arange(4 * 3) + 10).reshape((1, -1, 4)).to(torch.float32),)
    _dynamic = {"x": {0: torch.export.Dim("batch")}}


class AtenIndexPut3D_1(torch.nn.Module):

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


class AtenIndexPut3D_2(torch.nn.Module):

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
        return y

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
