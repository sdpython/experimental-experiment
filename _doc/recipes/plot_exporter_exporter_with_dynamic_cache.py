"""
.. _l-plot-torch-export-with-dynamic-cache-201:

Export a model using a custom type as input
===========================================

We will a class used in many model: :class:`transformers.cache_utils.DynamicCache`.

First try: it fails
+++++++++++++++++++
"""

import torch
from onnx_diagnostic.helpers.cache_helper import make_dynamic_cache


class ModelTakingDynamicCacheAsInput(torch.nn.Module):
    def forward(self, x, dc):
        kc = torch.cat(dc.key_cache, axis=1)
        vc = torch.cat(dc.value_cache, axis=1)
        y = (kc + vc).sum(axis=2, keepdim=True)
        return x + y


# %%
# Let's check the model runs.

x = torch.randn(3, 8, 7, 1)
cache = make_dynamic_cache([(torch.ones((3, 8, 5, 6)), (torch.ones((3, 8, 5, 6)) * 2))])

model = ModelTakingDynamicCacheAsInput()
expected = model(x, cache)

print(expected.shape)

# %%
# Let's check it works with other shapes.

x = torch.randn(4, 8, 7, 1)
cache = make_dynamic_cache([(torch.ones((4, 8, 11, 6)), (torch.ones((4, 8, 11, 6)) * 2))])

model = ModelTakingDynamicCacheAsInput()
expected = model(x, cache)

print(expected.shape)

# %%
# Let's export.

ep = torch.export.export(model, (x, cache))
print(ep.graph)

# %%
# With dynamic shapes now.


batch = torch.export.Dim("batch", min=1, max=1024)
clength = torch.export.Dim("clength", min=1, max=1024)

try:
    ep = torch.export.export(
        model,
        (x, cache),
        dynamic_shapes=({0: batch}, [[{0: batch, 2: clength}], [{0: batch, 2: clength}]]),
    )
    print(ep.graph)
    failed = False
except Exception as e:
    print("FAILS:", e)
    failed = True

# %%
# If it failed, let's understand why.

if failed:

    class Model(torch.nn.Module):
        def forward(self, dc):
            kc = dc.key_cache[0]
            vc = dc.value_cache[0]
            return kc + vc

    ep = torch.export.export(
        Model(),
        (cache,),
        dynamic_shapes={"dc": [[{0: batch, 2: clength}], [{0: batch, 2: clength}]]},
    )
    for node in ep.graph.nodes:
        print(f"{node.name} -> {node.meta.get('val', '-')}")
        # it prints out ``dc_key_cache_0 -> FakeTensor(..., size=(4, 8, 11, 6))``
        # but it should be ``dc_key_cache_0 -> FakeTensor(..., size=(s0, 8, s1, 6))``
