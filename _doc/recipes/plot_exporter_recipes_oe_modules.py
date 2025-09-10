"""
.. _l-plot-exporter-recipes-onnx-exporter-modules:

torch.onnx.export and submodules from LLMs
==========================================

**Incomplete**

Big models are hard to read once converted into onnx.
Let's see how to improve their readibility.
The code is inspired from
`LLM from scratch with Pytorch
<https://medium.com/@msouza.os/llm-from-scratch-with-pytorch-9f21808c6319>`_.

A simple LLM
++++++++++++

All comments were removed from the code to make it less verbose.
A few fixes were applied to the original code.
"""

import onnx
from onnx.printer import to_text
import torch
from onnxruntime import InferenceSession


class Embedding(torch.nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.pe = torch.nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        word_emb = self.embedding(x)
        word_pe = self.pe(x)
        return word_emb + word_pe


class AttentionBlock(torch.nn.Module):

    def __init__(self, embedding_dim: int, context_size: int):
        super().__init__()
        self.query = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)

        ones = torch.ones(size=[context_size, context_size], dtype=torch.float)
        self.register_buffer(name="mask", tensor=torch.tril(input=ones))

    def forward(self, x):
        _B, T, C = x.size()

        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        qk = query @ key.transpose(-2, -1) * C**-0.5
        attention = qk.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        attention = torch.nn.functional.softmax(input=attention, dim=-1)

        out = attention @ value
        return out


class MultiAttentionBlock(torch.nn.Module):

    def __init__(self, embedding_dim: int, num_heads: int, context_size: int):
        super().__init__()
        self.attention = torch.nn.ModuleList(
            modules=[AttentionBlock(embedding_dim, context_size) for _ in range(num_heads)]
        )
        self.linear = torch.nn.Linear(
            in_features=embedding_dim * num_heads, out_features=embedding_dim
        )

    def forward(self, x):
        out = torch.cat(tensors=[attention(x) for attention in self.attention], dim=-1)
        x = self.linear(out)
        return x


class FeedForward(torch.nn.Module):

    def __init__(self, embedding_dim: int, ff_dim: int):
        super().__init__()
        self.linear_1 = torch.nn.Linear(embedding_dim, ff_dim)
        self.relu = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(ff_dim, embedding_dim)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        return x


class DecoderLayer(torch.nn.Module):

    def __init__(self, embedding_dim: int, num_heads: int, context_size: int, ff_dim: int):
        super().__init__()
        self.attention = MultiAttentionBlock(embedding_dim, num_heads, context_size)
        self.feed_forward = FeedForward(embedding_dim, ff_dim)
        self.norm_1 = torch.nn.LayerNorm(normalized_shape=embedding_dim)
        self.norm_2 = torch.nn.LayerNorm(normalized_shape=embedding_dim)

    def forward(self, x):
        x_norm = self.norm_1(x)
        attention = self.attention(x_norm)
        attention = attention + x

        attention_norm = self.norm_2(attention)
        ff = self.feed_forward(attention_norm)
        ff = ff + attention

        return ff


class LLM(torch.nn.Module):

    def __init__(
        self,
        vocab_size: int = 1024,
        embedding_dim: int = 16,
        num_heads: int = 2,
        context_size: int = 256,
        ff_dim: int = 128,
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.decoder = DecoderLayer(embedding_dim, num_heads, context_size, ff_dim)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        y = self.decoder(x)
        return y


llm = LLM()
dim = (1, 30)
input_ids = torch.randint(0, 1024, dim).to(torch.int64)
y = llm(input_ids)

print(f"output: shape={y.shape}, min={y.min()}, max={y.max()}")

# %%
# First conversion to ONNX
# ++++++++++++++++++++++++
#
# The conversion relies on :func:`torch.export.export`.
# which gives:

ep = torch.export.export(llm, (input_ids,))
print(ep.graph)

# Then function :func:`torch.onnx.export` converts it into ONNX.

epo = torch.onnx.export(llm, (input_ids,), dynamo=True)
print(to_text(epo.model_proto))

# %%
# Let's check there is no discrepancy.

sess = InferenceSession(
    epo.model_proto.SerializeToString(), providers=["CPUExecutionProvider"]
)
feeds = dict(input_ids=input_ids.numpy())
got = sess.run(None, feeds)[0]

diff = torch.abs(y - torch.from_numpy(got)).max()
print(f"output: shape={got.shape}, min={got.min()}, max={got.max()}")
print(f"max discrepancy={diff}")

# %%
# Let's save the ONNX model.

onnx.save(epo.model_proto, "plot_exporter_recipes_c_modules.inlined.onnx")

# %%
# ONNX with submodules
# ++++++++++++++++++++
#
# Let's produce an ONNX model with submodules.
# Function :func:`to_onnx <experimental_experiment.torch_interpreter.to_onnx>`
# is calling the function :func:`torch.export.unflatten.unflatten`
# under the hood. The fx graph looks like the following.

ep = torch.export.export(llm, (input_ids,))
unflatten_ep = torch.export.unflatten(ep)
print(unflatten_ep.graph)

# %%
# The exported graph looks simpler and shows something like::
#
#   %decoder : [num_users=1] = call_module[target=decoder](args = (%embedding,), kwargs = {})
#
# It preserves the hierarchy but it does not necessarily preserves the signatures
# of the initial modules. That's was not one of our goals.
# The tricky part is module called (*embedding*) is not an instance ``Embedding``
# but an instance of `InterpreterModule
# <https://github.com/pytorch/pytorch/blob/main/torch/export/unflatten.py#L116>`_
# and contains the fx nodes contributing to the submodule and coming from the
# previous graph.
#
# And the ONNX graph.
