"""
.. _l-plot-exporter-recipes-custom-modules:

to_onnx and submodules from LLMs
================================

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
from onnx.inliner import inline_local_functions
from onnx_array_api.plotting.graphviz_helper import plot_dot
from onnx_array_api.reference import compare_onnx_execution
import torch
from onnxruntime import InferenceSession
from experimental_experiment.torch_interpreter import to_onnx
from experimental_experiment.helpers import pretty_onnx, max_diff
from experimental_experiment.xbuilder import OptimizationOptions


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
        B, T, C = x.size()

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

# %%
# Then function :func:`to_onnx <experimental_experiment.torch_interpreter.to_onnx>`
# converts it into ONNX.

onx = to_onnx(llm, (input_ids,))
print(pretty_onnx(onx))

# %%
# Let's check there is no discrepancy.

sess = InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
feeds = dict(input_ids=input_ids.numpy())
got = sess.run(None, feeds)[0]

diff = max_diff(y, got)
print(f"output: shape={got.shape}, min={got.min()}, max={got.max()}")
print(f"max discrepancy={diff['abs']}")

# %%
# Let's save the ONNX model.

onnx.save(onx, "plot_exporter_recipes_c_modules.inlined.onnx")

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
# Now the ONNX graph.

onx_module = to_onnx(llm, (input_ids,), export_modules_as_functions=True)
print(pretty_onnx(onx_module))

# %%
# We check again there is no new discrepancies.

sess = InferenceSession(onx_module.SerializeToString(), providers=["CPUExecutionProvider"])
feeds = dict(input_ids=input_ids.numpy())
got = sess.run(None, feeds)[0]

diff = max_diff(y, got)
print(f"output: shape={got.shape}, min={got.min()}, max={got.max()}")
print(f"max discrepancy={diff['abs']}")

# %%
# Let's save the ONNX model.

onnx.save(onx_module, "plot_exporter_recipes_c_modules.module.onnx")

# %%
# And visually.

plot_dot(onx_module)

# %%
# Inlining
# ++++++++
#
# The ONNX graph can still be inline after this.

onx_inlined = inline_local_functions(onx_module)
print(pretty_onnx(onx_inlined))

# %%
# Optimizations
# +++++++++++++
#
# The ONNX graph produced by the exporter without any optimization is very verbose
# and less efficient. That's why some optimizations are made to the model by default.
# It is also possible to introduce kernels implemented in :epkg:`onnxruntime`.
# Let's how it goes.

onx_optimized = to_onnx(
    llm,
    (input_ids,),
    options=OptimizationOptions(
        patterns="default+onnxruntime", constant_folding=True, verbose=2
    ),
)
print(pretty_onnx(onx_optimized))

# %%
# This shows a kernel ``FusedMatMul[com.microsoft]`` which implement a kernel equivalent Gemm
# but working for any tensors, not only 2D.
# How does it work on the model which keeps exports the moduels as local functions?
# The optimizer optimizes every local function independantly.
# We reduce the verbosity...

onx_module_optimized = to_onnx(
    llm,
    (input_ids,),
    options=OptimizationOptions(patterns="default+onnxruntime", constant_folding=True),
    export_modules_as_functions=True,
)
print(pretty_onnx(onx_module_optimized))

# %%
# It seems to be working as well on this simple case even though the optimizers were
# not tested on such models. However, keeping the submodule information might be useful
# to implement optimizer for a fmaily of models sharing the same components.
#
# Optimizations for CUDA
# ++++++++++++++++++++++
#
# The optimizer may have a different behaviour knowning the model is running on CUDA.
# It may use different kernels and do different optimization if needed.
# That may not be the good place to do it as the runtime may choose to run one kernel on CPU,
# another one on CUDA. The current optimization does not know that and
# is not able to decide which provider would be more useful for some kernels.
# This coudl even be decided at runtime.

onx_cuda_optimized = to_onnx(
    llm,
    (input_ids,),
    options=OptimizationOptions(
        patterns="default+onnxruntime", constant_folding=True, verbose=2, processor="CUDA"
    ),
)
print(pretty_onnx(onx_cuda_optimized))


# %%
# Comparison optimized and not optimized?
# +++++++++++++++++++++++++++++++++++++++
#
# The following tools is trying to match the node and shape inference
# from two models. If they are not too different, the functions
# is able to find out the differences. We can use to see
# which operators were fused into bigger ones only implemented by
# :epkg:`onnxruntime`.

res1, res2, align, dc = compare_onnx_execution(onx, onx_optimized, verbose=1)
print("------------")
text = dc.to_str(res1, res2, align)
print(text)

# %%
# The conversion should handle dynamic shapes as well as the input sequence
# can be of any length. But that's a topic for another example.
