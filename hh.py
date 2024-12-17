import torch


class Model(torch.nn.Module):
    def forward(self, x, y, z):
        y1 = y[: -x.shape[0], : -x.shape[1]]
        return z * y1


x, y, z = torch.rand((4, 5)), torch.rand((5, 6)), torch.rand((7, 8))
Model()(x, y, z)

d1 = torch.export.Dim("d1")
d2 = torch.export.Dim("d2")
d3 = torch.export.Dim("d3")
d4 = torch.export.Dim("d4")
ds = ({0: d1, 1: d2}, {0: d1 + 1, 1: d2 + 1}, {0: d3, 1: d4})
print(torch.export.export(Model(), (x, y, z), dynamic_shapes=ds))
