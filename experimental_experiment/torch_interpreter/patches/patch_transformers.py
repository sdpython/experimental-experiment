from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch


@dataclass
class patched_AttentionMaskConverter:
    """
    Patches
    ``transformers.modeling_attn_mask_utils.AttentionMaskConverter._make_causal_mask``.
    """

    @staticmethod
    def _make_causal_mask(
        input_ids_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        past_key_values_length: int = 0,
        sliding_window: Optional[int] = None,
    ):
        """Patched method."""
        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)

        mask = mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat(
                [
                    torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device),
                    mask,
                ],
                dim=-1,
            )

        if sliding_window is not None:
            diagonal = past_key_values_length - sliding_window - 1

            context_mask = torch.tril(
                torch.ones_like(mask, dtype=torch.bool), diagonal=diagonal
            )
            # In this case, the current implementation of torch fails (17/12/2024).
            # Try model Phi-3.5-Mini-Instruct.
            mask = mask.masked_fill(context_mask, torch.finfo(dtype).min)

        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


class patched_DynamicCache:
    """
    Removes the dependency on :class:`torch.nn.Module`
    from :class:`transformers.cache_utils.DynamicCache`.
    """

    def __init__(self, num_hidden_layers: Optional[int] = None) -> None:
        self._seen_tokens = 0
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(
                f"Cache only has {len(self)} layers, "
                f"attempted to access layer with index {layer_idx}"
            )

    def __iter__(self):
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        return len(self.key_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the number of seen tokens
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
            # elif (
            #    len(self.key_cache[layer_idx]) == 0
            # ):  # fills previously skipped layers; checking for tensor causes errors
            #    self.key_cache[layer_idx] = key_states
            #    self.value_cache[layer_idx] = value_states
            else:
                self.key_cache[layer_idx] = torch.cat(
                    [self.key_cache[layer_idx], key_states], dim=-2
                )
                self.value_cache[layer_idx] = torch.cat(
                    [self.value_cache[layer_idx], value_states], dim=-2
                )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if not self.key_cache:
            return 0
        assert layer_idx < len(
            self.key_cache
        ), f"Unexpected layer_idx={layer_idx}, len(key_cache)={len(self.key_cache)}"
        return self.key_cache[layer_idx].shape[-2]

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        return self.get_seq_length(layer_idx)

    def get_max_cache_shape(self) -> Optional[int]:
        return None

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(
        cls,
        past_key_values: Optional[Tuple[Tuple["torch.Tensor"]]] = None,
        num_hidden_layers: Optional[int] = None,
    ) -> "transformers.cache_utils.DynamicCache":  # noqa: F821
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache

    def crop(self, max_length: int):
        # In case it is negative
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)

        if self.get_seq_length() <= max_length:
            return

        self._seen_tokens = max_length
        for idx in range(len(self.key_cache)):
            if self.key_cache[idx] != []:
                self.key_cache[idx] = self.key_cache[idx][..., :max_length, :]
                self.value_cache[idx] = self.value_cache[idx][..., :max_length, :]

    def batch_split(
        self, full_batch_size: int, split_size: int, num_hidden_layers: Optional[int] = None
    ) -> List["transformers.cache_utils.DynamicCache"]:  # noqa: F821
        out = []
        for i in range(0, full_batch_size, split_size):
            current_split = patched_DynamicCache()
            current_split._seen_tokens = self._seen_tokens
            current_split.key_cache = [tensor[i : i + split_size] for tensor in self.key_cache]
            current_split.value_cache = [
                tensor[i : i + split_size] for tensor in self.value_cache
            ]
            out.append(current_split)
        return out

    @classmethod
    def from_batch_splits(
        cls,
        splits: List["transformers.cache_utils.DynamicCache"],  # noqa: F821
        num_hidden_layers: Optional[int] = None,
    ) -> "transformers.cache_utils.DynamicCache":  # noqa: F821
        cache = cls()
        for idx in range(len(splits[0])):
            key_cache = [
                current.key_cache[idx] for current in splits if current.key_cache[idx] != []
            ]
            value_cache = [
                current.value_cache[idx]
                for current in splits
                if current.value_cache[idx] != []
            ]
            if key_cache != []:
                layer_keys = torch.cat(key_cache, dim=0)
                layer_values = torch.cat(value_cache, dim=0)
                cache.update(layer_keys, layer_values, idx)
        return cache

    def batch_repeat_interleave(self, repeats: int):
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx].repeat_interleave(
                repeats, dim=0
            )
            self.value_cache[layer_idx] = self.value_cache[layer_idx].repeat_interleave(
                repeats, dim=0
            )

    def batch_select_indices(self, indices: torch.Tensor):
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][indices, ...]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][indices, ...]
