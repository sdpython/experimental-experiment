"""
More complex helpers used in unit tests.
"""

import contextlib
import io
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from onnx import ModelProto, save
from .helpers import pretty_onnx


def to_numpy(tensor: "torch.Tensor"):  # noqa: F821
    """
    Convets a torch tensor to numy.
    """
    try:
        return tensor.numpy()
    except TypeError:
        # We try with ml_dtypes
        pass

    import ml_dtypes
    import torch

    conv = {torch.bfloat16: ml_dtypes.bfloat16}
    assert tensor.dtype in conv, f"Unsupported type {tensor.dtype}, not in {conv}"
    return tensor.to(torch.float32).numpy().astype(conv[tensor.dtype])


def check_model_ort(
    onx: ModelProto,
    providers: Optional[Union[str, List[str]]] = None,
    dump_file: Optional[str] = None,
) -> "onnxruntime.InferenceSession":  # noqa: F821
    """
    Loads a model with onnxruntime.

    :param onx: ModelProto
    :param providers: list of providers, None fur CPU, cpu for CPU, cuda for CUDA
    :param dump_file: if not empty, dumps the model into this file if
        an error happened
    :return: InferenceSession
    """
    from onnxruntime import InferenceSession

    if providers is None or providers == "cpu":
        providers = ["CPUExecutionProvider"]
    elif not isinstance(providers, list) and providers.startswith("cuda"):
        device_id = 0 if ":" not in providers else int(providers.split(":")[1])
        providers = [
            ("CUDAExecutionProvider", {"device_id": device_id}),
            ("CPUExecutionProvider", {}),
        ]

    if isinstance(onx, str):
        try:
            return InferenceSession(onx, providers=providers)
        except Exception as e:
            import onnx

            if dump_file:
                save(onx, dump_file)

            raise AssertionError(  # noqa: B904
                f"onnxruntime cannot load the model "
                f"due to {e}\n{pretty_onnx(onnx.load(onx))}"
            )
        return
    try:
        return InferenceSession(onx.SerializeToString(), providers=providers)
    except Exception as e:
        if dump_file:
            save(onx, dump_file)
        raise AssertionError(  # noqa: B904
            f"onnxruntime cannot load the modeldue to {e}\n{pretty_onnx(onx)}"
        )


def export_to_onnx(
    model: Any,
    *args: List[Any],
    verbose: int = 0,
    return_builder: bool = False,
    torch_script: bool = True,
    target_opset: int = 18,
    prefix: Optional[str] = None,
    rename_inputs: bool = False,
    optimize: Union[str, bool] = True,
    folder: Optional[str] = "dump_test",
    export_options: Optional["ExportOptions"] = None,  # noqa: F821
) -> Dict[str, Union[str, ModelProto, "GraphBuilder"]]:  # noqa: F821
    """
    Exports a model to ONNX.

    :param model: model to export
    :param args: arguments
    :param verbose: verbosity
    :param return_builder: returns the builder
    :param torch_script: export with torch.script as well
    :param target_opset: opset to export into
    :param prefix: prefix to choose to export into
    :param rename_inputs: rename the inputs into ``input_{i}``
    :param optimize: enable, disable optimizations of pattern to test
    :param folder: where to dump the model, creates it if it does not exist
    :param export_options: see :class:`ExportOptions
        <experimental_experiment.torch_interpreter.ExportOptions>`
    :return: dictionary with ModelProto, builder, filenames
    """
    from .xbuilder import OptimizationOptions
    from .torch_interpreter import to_onnx

    ret = {}
    if torch_script and prefix is not None:
        import torch

        filename = f"{prefix}.onnx"
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(model, args, filename, input_names=["input"])
            ret["torch.script"] = filename

    if isinstance(optimize, str):
        options = OptimizationOptions(verbose=verbose, patterns=optimize)
    else:
        options = OptimizationOptions(verbose=verbose)
    onx = to_onnx(
        model,
        tuple(args),
        input_names=[f"input{i}" for i in range(len(args))] if rename_inputs else None,
        options=options,
        verbose=verbose,
        return_builder=return_builder,
        optimize=optimize,
        export_options=export_options,
    )
    ret["proto"] = onx
    if prefix is not None:
        filename = f"{prefix}.custom.onnx"
        if folder:
            if not os.path.exists(folder):
                os.makedirs(folder)
            filename = os.path.join(folder, filename)
        with open(filename, "wb") as f:
            f.write((onx[0] if return_builder else onx).SerializeToString())
        ret["custom"] = filename
    return ret


def dummy_llm(
    cls_name: Optional[str] = None,
    dynamic_shapes: bool = False,
) -> Tuple["torch.nn.Module", Tuple["torch.Tensor", ...]]:  # noqa: F821
    """
    Creates a dummy LLM for test purposes.

    :param cls_name: None for whole model or a piece of it
    :param dynamic_shapes: returns dynamic shapes as well

    .. runpython::
        :showcode:

        from experimental_experiment.torch_test_helper import dummy_llm
        print(dummy_llm())
    """
    import torch

    class Embedding(torch.nn.Module):
        def __init__(self, vocab_size: int = 1024, embedding_dim: int = 16):
            super().__init__()
            self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
            self.pe = torch.nn.Embedding(vocab_size, embedding_dim)

        def forward(self, x):
            word_emb = self.embedding(x)
            word_pe = self.pe(x)
            return word_emb + word_pe

    class AttentionBlock(torch.nn.Module):

        def __init__(self, embedding_dim: int = 16, context_size: int = 256):
            super().__init__()
            self.query = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
            self.key = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
            self.value = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
            # torch.nn.Buffer are not fully handled by symbolic tracing
            # Buffer(...)[:Prowy()] is not working
            self.mask = torch.nn.Parameter(
                torch.tril(
                    input=torch.ones(size=[context_size, context_size], dtype=torch.float)
                )
            )

        def forward(self, x):
            B, T, C = x.shape

            query = self.query(x)
            key = self.key(x)
            value = self.value(x)

            qk = query @ key.transpose(-2, -1) * C**-0.5
            attention = qk.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
            attention = torch.nn.functional.softmax(input=attention, dim=-1)

            out = attention @ value
            return out

    class MultiAttentionBlock(torch.nn.Module):

        def __init__(
            self, embedding_dim: int = 16, num_heads: int = 2, context_size: int = 256
        ):
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

        def __init__(self, embedding_dim: int = 16, ff_dim: int = 128):
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

        def __init__(
            self,
            embedding_dim: int = 16,
            num_heads: int = 2,
            context_size: int = 256,
            ff_dim: int = 128,
        ):
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

    if cls_name in (None, "LLM"):
        dec = LLM()
        x = torch.randint(0, 1024, (2 if dynamic_shapes else 1, 30)).to(torch.int64)
        dec(x)
        if dynamic_shapes:
            dyn = {
                "input_ids": {
                    0: torch.export.Dim("batch", min=1, max=1024),
                    1: torch.export.Dim("length", min=1, max=255),
                }
            }
            return dec, (x,), dyn
        return dec, (x,)

    if cls_name == "DecoderLayer":
        LLM()(torch.randint(0, 1024, (2 if dynamic_shapes else 1, 30)).to(torch.int64))

        dec = DecoderLayer()
        x = Embedding()(
            torch.randint(0, 1024, (2 if dynamic_shapes else 1, 30)).to(torch.int64)
        )
        dec(x)
        if dynamic_shapes:
            dyn = {
                "x": {
                    0: torch.export.Dim("batch", min=1, max=1024),
                    1: torch.export.Dim("length", min=1, max=255),
                }
            }
            return dec, (x,), dyn
        return dec, (x,)

    if cls_name == "MultiAttentionBlock":
        dec = MultiAttentionBlock()
        x = torch.rand(2 if dynamic_shapes else 1, 30, 16).to(torch.float32)
        dec(x)
        if dynamic_shapes:
            dyn = {
                "x": {
                    0: torch.export.Dim("batch", min=1, max=1024),
                    1: torch.export.Dim("length", min=1, max=255),
                }
            }
            return dec, (x,), dyn
        return dec, (x,)

    if cls_name == "AttentionBlock":
        dec = AttentionBlock()
        x = torch.rand(2 if dynamic_shapes else 1, 30, 16).to(torch.float32)
        dec(x)
        if dynamic_shapes:
            dyn = {
                "x": {
                    0: torch.export.Dim("batch", min=1, max=1024),
                    1: torch.export.Dim("length", min=1, max=255),
                }
            }
            return dec, (x,), dyn
        return dec, (x,)

    raise NotImplementedError(f"cls_name={cls_name}")
