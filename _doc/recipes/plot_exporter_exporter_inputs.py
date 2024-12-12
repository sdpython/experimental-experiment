"""
.. _l-plot-exporter-nn_modules_inputs:

Do no use Module as inputs!
===========================

This continues example :ref:`l-plot-torch-export-with-dynamic-cache-201`.

Custom classes are working fine
+++++++++++++++++++++++++++++++

``DynamicCache`` is replica of :class:`transformers.cache_utils.DynamicCache`
but it does not inherits from :class:`transformers.cache_utils.Cache`.
"""

from typing import Any, Dict, List, Optional, Tuple
import torch


class DynamicCache:
    def __init__(self, num_hidden_layers: Optional[int] = None) -> None:
        super().__init__()
        self._seen_tokens = (
            0  # Used in `generate` to keep tally of how many tokens the cache has seen
        )
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if key_states is not None:
            if len(self.key_cache) <= layer_idx:
                # There may be skipped layers, fill them with empty lists
                for _ in range(len(self.key_cache), layer_idx):
                    self.key_cache.append([])
                    self.value_cache.append([])
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            elif (
                len(self.key_cache[layer_idx]) == 0
            ):  # fills previously skipped layers; checking for tensor causes errors
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
            else:
                self.key_cache[layer_idx] = torch.cat(
                    [self.key_cache[layer_idx], key_states], dim=-2
                )
                self.value_cache[layer_idx] = torch.cat(
                    [self.value_cache[layer_idx], value_states], dim=-2
                )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        is_empty_layer = (
            len(self.key_cache) == 0  # no cache in any layer
            or len(self.key_cache)
            <= layer_idx  # skipped `layer_idx` and hasn't run a layer with cache after it
            or len(self.key_cache[layer_idx]) == 0  # the layer has no cache
        )
        layer_seq_length = self.key_cache[layer_idx].shape[-2] if not is_empty_layer else 0
        return layer_seq_length


##############################
# A model uses the class we introduced.


class ModelTakingDynamicCacheAsInput(torch.nn.Module):
    def forward(self, x, dc):
        kc = torch.cat(dc.key_cache, axis=1)
        vc = torch.cat(dc.value_cache, axis=1)
        length = dc.get_seq_length() if dc is not None else 0
        ones = torch.zeros(
            (
                dc.key_cache[0].shape[0],
                dc.key_cache[0].shape[1],
                length,
                dc.key_cache[0].shape[-1],
            )
        )
        catones = torch.cat((kc + vc, ones), dim=1)
        y = catones.sum(axis=2, keepdim=True)
        return x + y


###########################
# Let's check the model runs.

x = torch.randn(3, 8, 7, 1)
cache = DynamicCache(1)
cache.update(torch.ones((3, 8, 5, 6)), (torch.ones((3, 8, 5, 6)) * 2), 0)

model = ModelTakingDynamicCacheAsInput()
expected = model(x, cache)

print(expected.shape)

###########################
# Let's check it works with others shapes.

x = torch.randn(4, 8, 7, 1)
cache = DynamicCache(1)
cache.update(torch.ones((4, 8, 11, 6)), (torch.ones((4, 8, 11, 6)) * 2), 0)

model = ModelTakingDynamicCacheAsInput()
expected = model(x, cache)

print(expected.shape)

##########################
# Let's export after serialization functions were registered as shown in
# :ref:`l-plot-torch-export-with-dynamic-cache-201`


def flatten_dynamic_cache(
    dynamic_cache: DynamicCache,
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
) -> DynamicCache:
    cache = DynamicCache()
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
    DynamicCache,
    flatten_dynamic_cache,
    unflatten_dynamic_cache,
    serialized_type_name=f"{DynamicCache.__module__}.{DynamicCache.__name__}",
    flatten_with_keys_fn=flatten_with_keys_dynamic_cache,
)
torch.fx._pytree.register_pytree_flatten_spec(
    DynamicCache, lambda x, _: [x.key_cache, x.value_cache]
)


########################################
# Let's export with dynamic shapes.

batch = torch.export.Dim("batch", min=1, max=1024)
clength = torch.export.Dim("clength", min=1, max=1024)

ep = torch.export.export(
    model,
    (x, cache),
    dynamic_shapes=({0: batch}, [[{0: batch, 2: clength}], [{0: batch, 2: clength}]]),
)
print(ep)


#####################################
# We remove the changes for pytorch.

torch.utils._pytree.SUPPORTED_NODES.pop(DynamicCache)
torch.fx._pytree.SUPPORTED_NODES.pop(DynamicCache)
torch.fx._pytree.SUPPORTED_NODES_EXACT_MATCH.pop(DynamicCache)

##########################################
# Everything looks fine but now...
#
# DynamicCache(torch.nn.Module)
# +++++++++++++++++++++++++++++
#
# That's the only change we make.
# Everything else is the same.


class DynamicCache(torch.nn.Module):
    def __init__(self, num_hidden_layers: Optional[int] = None) -> None:
        super().__init__()
        self._seen_tokens = (
            0  # Used in `generate` to keep tally of how many tokens the cache has seen
        )
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if key_states is not None:
            if len(self.key_cache) <= layer_idx:
                # There may be skipped layers, fill them with empty lists
                for _ in range(len(self.key_cache), layer_idx):
                    self.key_cache.append([])
                    self.value_cache.append([])
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            elif (
                len(self.key_cache[layer_idx]) == 0
            ):  # fills previously skipped layers; checking for tensor causes errors
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
            else:
                self.key_cache[layer_idx] = torch.cat(
                    [self.key_cache[layer_idx], key_states], dim=-2
                )
                self.value_cache[layer_idx] = torch.cat(
                    [self.value_cache[layer_idx], value_states], dim=-2
                )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        is_empty_layer = (
            len(self.key_cache) == 0  # no cache in any layer
            or len(self.key_cache)
            <= layer_idx  # skipped `layer_idx` and hasn't run a layer with cache after it
            or len(self.key_cache[layer_idx]) == 0  # the layer has no cache
        )
        layer_seq_length = self.key_cache[layer_idx].shape[-2] if not is_empty_layer else 0
        return layer_seq_length


##############################
# A model uses the class we introduced.


class ModelTakingDynamicCacheAsInput(torch.nn.Module):
    def forward(self, x, dc):
        kc = torch.cat(dc.key_cache, axis=1)
        vc = torch.cat(dc.value_cache, axis=1)
        length = dc.get_seq_length() if dc is not None else 0
        ones = torch.zeros(
            (
                dc.key_cache[0].shape[0],
                dc.key_cache[0].shape[1],
                length,
                dc.key_cache[0].shape[-1],
            )
        )
        catones = torch.cat((kc + vc, ones), dim=1)
        y = catones.sum(axis=2, keepdim=True)
        return x + y


###########################
# Let's check the model runs.

x = torch.randn(3, 8, 7, 1)
cache = DynamicCache(1)
cache.update(torch.ones((3, 8, 5, 6)), (torch.ones((3, 8, 5, 6)) * 2), 0)

model = ModelTakingDynamicCacheAsInput()
expected = model(x, cache)

print(expected.shape)

###########################
# Let's check it works with others shapes.

x = torch.randn(4, 8, 7, 1)
cache = DynamicCache(1)
cache.update(torch.ones((4, 8, 11, 6)), (torch.ones((4, 8, 11, 6)) * 2), 0)

model = ModelTakingDynamicCacheAsInput()
expected = model(x, cache)

print(expected.shape)

##########################
# Let's export after serialization functions were registered as shown in
# :ref:`l-plot-torch-export-with-dynamic-cache-201`


def flatten_dynamic_cache(
    dynamic_cache: DynamicCache,
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
) -> DynamicCache:
    cache = DynamicCache()
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
    DynamicCache,
    flatten_dynamic_cache,
    unflatten_dynamic_cache,
    serialized_type_name=f"{DynamicCache.__module__}.{DynamicCache.__name__}",
    flatten_with_keys_fn=flatten_with_keys_dynamic_cache,
)
torch.fx._pytree.register_pytree_flatten_spec(
    DynamicCache, lambda x, _: [x.key_cache, x.value_cache]
)


########################################
# Let's export with dynamic shapes.

batch = torch.export.Dim("batch", min=1, max=1024)
clength = torch.export.Dim("clength", min=1, max=1024)

try:
    ep = torch.export.export(
        model,
        (x, cache),
        dynamic_shapes=({0: batch}, [[{0: batch, 2: clength}], [{0: batch, 2: clength}]]),
    )
    print(ep)
except Exception as e:
    print(f"It did not work: {e}")


#####################################
# We remove the changes for pytorch.

torch.utils._pytree.SUPPORTED_NODES.pop(DynamicCache)
torch.fx._pytree.SUPPORTED_NODES.pop(DynamicCache)
torch.fx._pytree.SUPPORTED_NODES_EXACT_MATCH.pop(DynamicCache)
