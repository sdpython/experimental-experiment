"""
.. _l-plot-torch-export-untrained-tinyllm:

Check the exporter on a dummy from HuggingFace
==============================================

Every conversion task must be tested on a large scale. One huge source
of model is :epkg:`HuggingFace`. We focus on the model
`Tiny-LLM <https://huggingface.co/arnir0/Tiny-LLM>`_.
To avoid downloading any weigths, we write a function creating a
random model based on the same architecture.

Guess the cache dimension
+++++++++++++++++++++++++

The first step is to guess the dummy inputs.
Let's use the true model for that.
We use the dummy example from the model page.
"""

from typing import Any, Dict, List, Tuple
import packaging.version as pv
import torch
import transformers


MODEL_NAME = "arnir0/Tiny-LLM"
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# %%
# We rewrite the forward method to print the cache dimension.


def string_inputs(args, kwargs):
    def _cache(a):
        if len(a.key_cache):
            return f"n_caches={len(a.key_cache)}, shape={a.key_cache[0].shape}"
        return f"n_caches={len(a.key_cache)}"

    for a in args:
        if isinstance(a, transformers.cache_utils.DynamicCache):
            return _cache(a)
    for k, a in kwargs.items():
        if isinstance(a, transformers.cache_utils.DynamicCache):
            return f"{k}={_cache(a)}"
    return "no_cache"


def _forward_(*args, _f=None, **kwargs):
    assert _f is not None
    if not torch.compiler.is_exporting():
        print(string_inputs(args, kwargs))
    return _f(*args, **kwargs)


keep_model_forward = model.forward
model.forward = lambda *args, _f=keep_model_forward, **kwargs: _forward_(
    *args, _f=_f, **kwargs
)

# %%
# Let's run the model.
prompt = "Continue: it rains..."
inputs = tokenizer.encode(prompt, return_tensors="pt")

outputs = model.generate(
    inputs, max_length=50, temperature=1, top_k=50, top_p=0.95, do_sample=True
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)

# %%
# Let's restore the forward as it was.
model.forward = keep_model_forward

# %%
# The model creation
# ++++++++++++++++++
#
# Let's create an untrained model.

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


def get_tiny_llm(
    batch_size: int = 2,
    input_cache: bool = True,
    common_dynamic_shapes: bool = True,
    dynamic_rope: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Gets a non initialized model.

    :param batch_size: batch size
    :param input_cache: generate data for this iteration with or without cache
    :param kwargs: to overwrite the configuration, example ``num_hidden_layers=1``
    :param common_dynamic_shapes: if True returns dynamic shapes as well
    :param dynamic_rope: use dynamic rope (see :class:`transformers.LlamaConfig`)
    :return: dictionary
    """
    import transformers

    config = {
        "architectures": ["LlamaForCausalLM"],
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": 192,
        "initializer_range": 0.02,
        "intermediate_size": 1024,
        "max_position_embeddings": 1024,
        "model_type": "llama",
        "num_attention_heads": 2,
        "num_hidden_layers": 1,
        "num_key_value_heads": 1,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-05,
        "rope_scaling": {"rope_type": "dynamic", "factor": 10.0} if dynamic_rope else None,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
        "transformers_version": "4.31.0.dev0",
        "use_cache": True,
        "vocab_size": 32000,
    }

    config.update(**kwargs)
    conf = transformers.LlamaConfig(**config)
    model = transformers.LlamaForCausalLM(conf)
    model.eval()

    # now the inputs
    cache_last_dim = 96
    sequence_length = 30
    sequence_length2 = 3
    num_key_value_heads = 1
    max_token_id = config["vocab_size"] - 1
    n_layers = config["num_hidden_layers"]

    batch = torch.export.Dim("batch", min=1, max=1024)
    seq_length = torch.export.Dim("seq_length", min=1, max=4096)
    cache_length = torch.export.Dim("cache_length", min=1, max=4096)

    shapes = {
        "input_ids": {0: batch, 1: seq_length},
        "attention_mask": {
            0: batch,
            1: torch.export.Dim.DYNAMIC,  # cache_length + seq_length
        },
        "past_key_values": [
            [{0: batch, 2: cache_length} for _ in range(n_layers)],
            [{0: batch, 2: cache_length} for _ in range(n_layers)],
        ],
    }
    inputs = dict(
        input_ids=torch.randint(0, max_token_id, (batch_size, sequence_length2)).to(
            torch.int64
        ),
        attention_mask=torch.ones((batch_size, sequence_length + sequence_length2)).to(
            torch.int64
        ),
        past_key_values=make_dynamic_cache(
            [
                (
                    torch.randn(
                        batch_size, num_key_value_heads, sequence_length, cache_last_dim
                    ),
                    torch.randn(
                        batch_size, num_key_value_heads, sequence_length, cache_last_dim
                    ),
                )
                for i in range(n_layers)
            ]
        ),
    )
    return dict(inputs=inputs, model=model, dynamic_shapes=shapes)


# %%
# Let's get the model, inputs and dynamic shapes.

experiment = get_tiny_llm()
model, inputs, dynamic_shapes = (
    experiment["model"],
    experiment["inputs"],
    experiment["dynamic_shapes"],
)

# %% Let's run it.
expected_output = model(**inputs)
print("result type", type(expected_output))

# %%
# It works.
#
# ExportedProgram
# +++++++++++++++

try:
    ep = torch.export.export(model, (), inputs, dynamic_shapes=dynamic_shapes)
    print("It worked:")
    print(ep)
except Exception as e:
    print("It failed:", e)
