from typing import List, Tuple
import packaging.version as pv
import torch
import transformers
import transformers.cache_utils

if pv.Version(transformers.__version__) > pv.Version("4.49.99999"):

    def make_dynamic_cache(
        key_value_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> transformers.cache_utils.DynamicCache:
        """
        Creates an instance of :class:`transformers.cache_utils.DynamicCache`.
        This version is valid for ``transformers >= 4.50``.

        :param key_value_pairs: list of pairs of (key, values)
        :return: :class:`transformers.cache_utils.DynamicCache`
        """
        return transformers.cache_utils.DynamicCache(key_value_pairs)

else:

    def make_dynamic_cache(
        key_value_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> transformers.cache_utils.DynamicCache:
        """
        Creates an instance of :class:`transformers.cache_utils.DynamicCache`.
        This version is valid for ``transformers < 4.50``.

        :param key_value_pairs: list of pairs of (key, values)
        :return: :class:`transformers.cache_utils.DynamicCache`
        """
        cache = transformers.cache_utils.DynamicCache(len(key_value_pairs))
        for i, (key, value) in enumerate(key_value_pairs):
            cache.update(key, value, i)
        return cache
