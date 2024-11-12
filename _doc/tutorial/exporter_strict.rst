.. _l-exporter-recipes:

================================================
Parameter torch.export.export(..., strict: bool)
================================================

The exporter relies on :func:`torch.export.export`. It exposes a parameter called
`strict: bool = True` (true by default).
The behaviour is different in some specific configuration.

struct=True
===========

torch.ops.higher_order.scan
+++++++++++++++++++++++++++

Not all signatures work with this mode.
Here is an example with scan.

.. runpython::
    :showcode:

    import torch

    def add(carry: torch.Tensor, y: torch.Tensor):
        next_carry = carry + y
        return [next_carry, next_carry]

    class ScanModel(torch.nn.Module):
        def forward(self, x):
            init = torch.zeros_like(x[0])
            carry, out = torch.ops.higher_order.scan(
                add, [init], [x], dim=0, reverse=False, additional_inputs=[]
            )
            return carry

    x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
    model = ScanModel()
    expected = model(x)
    print("------")
    print(expected, x.sum(axis=0))
    print("------")
    print(torch.export.export(model, (x,), strict=True).graph)

strict=False
===========

'from_node' missing in node.meta
++++++++++++++++++++++++++++++++

Every node of the obtained with ``strict=False`` has no key ``'from_node'``
in dictionary ``node.meta``. It is therefore difficult to trace where a parameter
is coming from unless this information is passed along when looking
into the submodules.

inplace x[..., i] = y
+++++++++++++++++++++

This expression cannot be captured with ``strict=False``.

.. runpython::
    :showcode:

    import torch

    class UpdateModel(torch.nn.Module):
        def forward(
            self, x: torch.Tensor, update: torch.Tensor, kv_index: torch.LongTensor
        ):
            x = x.clone()
            x[..., kv_index] = update
            return x

    example_inputs = (
        torch.ones((4, 4, 10)).to(torch.float32),
        (torch.arange(2) + 10).to(torch.float32).reshape((1, 1, 2)),
        torch.Tensor([1, 2]).to(torch.int32),
    )

    model = UpdateModel()

    try:
        torch.export.export(model, (x,), strict=False)
    except Exception as e:
        print(e)
