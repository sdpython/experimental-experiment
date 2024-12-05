"""
.. _l-plot-torch-export-with-dynamic-cache-201:

201: Export a model using a custom type as input
================================================

We will a class used in many model: :class:`transformers.cache_utils.DynamicCache`.

First try: it fails
+++++++++++++++++++
"""

from typing import Any, Dict, List, Tuple
import torch
import transformers


class ModelTakingDynamicCacheAsInput(torch.nn.Module):
    def forward(self, x, dc):
        kc = torch.cat(dc.key_cache, axis=1)
        vc = torch.cat(dc.value_cache, axis=1)
        y = (kc + vc).sum(axis=2, keepdim=True)
        return x + y


###########################
# Let's check the model runs.

x = torch.randn(3, 8, 7, 1)
cache = transformers.cache_utils.DynamicCache(1)
cache.update(torch.ones((3, 8, 5, 6)), (torch.ones((3, 8, 5, 6)) * 2), 0)

model = ModelTakingDynamicCacheAsInput()
expected = model(x, cache)

print(expected.shape)

###########################
# Let's check it works with others shapes.

x = torch.randn(4, 8, 7, 1)
cache = transformers.cache_utils.DynamicCache(1)
cache.update(torch.ones((4, 8, 11, 6)), (torch.ones((4, 8, 11, 6)) * 2), 0)

model = ModelTakingDynamicCacheAsInput()
expected = model(x, cache)

print(expected.shape)

##########################
# Let's export.

try:
    torch.export.export(model, (x, cache))
except Exception as e:
    print("export failed with", e)


###########################
# Register serialization of DynamicCache
# ++++++++++++++++++++++++++++++++++++++
#
# That's what needs to be done.
# Feel free to adapt it to your own class.
# The important informatin is we want to serialize
# two attributes ``key_cache`` and ``value_cache``.
# Both are list of tensors of the same size.


def flatten_dynamic_cache(
    dynamic_cache: transformers.cache_utils.DynamicCache,
) -> Tuple[List[Any], torch.utils._pytree.Context]:
    flat = [
        (k, getattr(dynamic_cache, k))
        for k in ["key_cache", "value_cache"]
        if hasattr(dynamic_cache, k)
    ]
    return [f[1] for f in flat], [f[0] for f in flat]


def unflatten_dynamic_cache(
    values: List[Any],
    context: torch.utils._pytree.Context,
    output_type=None,
) -> transformers.cache_utils.DynamicCache:
    cache = transformers.cache_utils.DynamicCache()
    values = dict(zip(context, values))
    for k, v in values.items():
        setattr(cache, k, v)
    return cache


def flatten_with_keys_dynamic_cache(d: Dict[Any, Any]) -> Tuple[
    List[Tuple[torch.utils._pytree.KeyEntry, Any]],
    torch.utils._pytree.Context,
]:
    values, context = flatten_dynamic_cache(d)
    return [(torch.utils._pytree.MappingKey(k), v) for k, v in zip(context, values)], context


torch.utils._pytree.register_pytree_node(
    transformers.cache_utils.DynamicCache,
    flatten_dynamic_cache,
    unflatten_dynamic_cache,
    serialized_type_name=f"{transformers.cache_utils.DynamicCache.__module__}.{transformers.cache_utils.DynamicCache.__name__}",
    flatten_with_keys_fn=flatten_with_keys_dynamic_cache,
)
torch.fx._pytree.register_pytree_flatten_spec(
    transformers.cache_utils.DynamicCache, lambda x, _: [x.key_cache, x.value_cache]
)


########################################
# Let's try to export again.
ep = torch.export.export(model, (x, cache))
print(ep.graph)

########################################
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

########################################
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
