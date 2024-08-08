import unittest
from experimental_experiment.ext_test_case import ExtTestCase, requires_cuda


class TestCustomOpsOnnxScript(ExtTestCase):
    @classmethod
    def setUpClass(cls):
        import onnxruntime  # noqa: F401

    @requires_cuda()
    def test_llama_sdpa_model_efficient(self):
        # see https://pytorch.org/tutorials/beginner/onnx/onnx_registry_tutorial.html
        # python -m pip install torch_ort
        # python -m torch_ort configure
        import contextlib
        import os
        import warnings
        from typing import Callable, List, Optional, Tuple
        import numpy as np
        import onnx
        import onnx.helper as oh
        from onnx import ModelProto
        import onnxscript
        import torch
        import torch.onnx

        def modify_onnx(model: ModelProto):
            # ScatterND + Aten ops
            print("[modify_onnx] START")
            for node in model.graph.node:
                if node.op_type == "ScatterND":
                    if len(node.attribute) == 0:
                        print("[modify_onnx] ScatterND, add reduction to add")
                        node.attribute.append(oh.make_attribute("reduction", "add"))
                    else:
                        red = node.attribute[0].s
                        if red != b"add":
                            print("[modify_onnx] ScatterND, change reduction to add")
                            del node.attribute[:]
                            node.attribute.append(oh.make_attribute("reduction", "add"))
                elif node.op_type == "ATen":
                    fname = None
                    for att in node.attribute:
                        if att.name == "operator":
                            fname = att.s
                    if fname == b"_scaled_dot_product_efficient_attention_backward":
                        print(
                            "[modify_onnx] ATen, delete last output for "
                            "_scaled_dot_product_efficient_attention_backward"
                        )
                        outputs = list(node.output)
                        del node.output[:]
                        outputs[-1] = ""
                        node.output.extend(outputs)
            print("[modify_onnx] DONE")
            return model

        @contextlib.contextmanager
        def dump_onnx(prefix: str, folder: Optional[str] = None, clean: bool = False):
            if folder:
                if not os.path.exists(folder):
                    os.makedirs(folder)
                if clean:
                    for f in os.listdir(folder):
                        ff = os.path.join(folder, f)
                        if os.path.isfile(ff):
                            os.remove(ff)
            else:
                assert not clean, "cleaning can only happen if folder is specified"

            value = os.environ.get("ONNXRT_DUMP_PATH", None)
            os.environ["ONNXRT_DUMP_PATH"] = os.path.join(folder, f"{prefix}_")

            try:
                yield
            finally:
                os.environ["ONNXRT_DUMP_PATH"] = value or ""

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from transformers import LlamaConfig
            from transformers.models.llama.modeling_llama import LlamaModel

        def ids_tensor(shape, vocab_size):
            total_dims = 1
            for dim in shape:
                total_dims *= dim

            values = []
            for _ in range(total_dims):
                values.append(np.random.randint(0, vocab_size - 1))

            return torch.tensor(data=values, dtype=torch.long).view(shape).contiguous()

        op = onnxscript.opset18
        aten_opset = onnxscript.values.Opset("org.pytorch.aten", 1)

        @onnxscript.script(aten_opset, default_opset=op)
        def scaled_dot_product_efficient_attention(
            query,
            key,
            value,
            attn_bias,
            compute_log_sumexp: bool,
            dropout_p: float,
            is_causal: bool,
        ):
            output, log_sumexp, philox_seed, philox_offset = aten_opset.ATen(
                query,
                key,
                value,
                attn_bias,
                compute_log_sumexp,
                dropout_p,
                is_causal,
                1.0,
                operator="_scaled_dot_product_efficient_attention",
            )
            return output, log_sumexp, philox_seed, philox_offset

        @onnxscript.script(aten_opset, default_opset=op)
        def scaled_dot_product_attention_backward(
            grad,
            query,
            key,
            value,
            attn_bias,
            output,
            logsumexp,
            philox_seed,
            philox_offset,
            dropout_p,
            grad_input_mask,
            is_causal,
        ):
            grad_query, grad_key, grad_value, grad_attn_bias = aten_opset.ATen(
                grad,  # 0
                query,  # 1
                key,  # 2
                value,  # 3
                attn_bias,  # 4
                output,  # 5
                logsumexp,  # 6
                philox_seed,  # 7
                philox_offset,  # 8
                dropout_p,  # 9
                grad_input_mask,  # 10
                is_causal,  # 11
                1.0,
                operator="_scaled_dot_product_efficient_attention_backward",
                # cpu_input_args=[],
            )
            return grad_query, grad_key, grad_value, grad_attn_bias

        def make_aot_ort(
            dynamic: bool = False,
            rewrite: bool = False,
            aten_conversion_changes: Optional[List[Tuple[Callable, str]]] = None,
            verbose: int = 0,
        ):
            import onnxruntime
            from torch.onnx import (
                OnnxRegistry,
                _OrtBackend as OrtBackend,
                _OrtBackendOptions as OrtBackendOptions,
                ExportOptions,
            )

            names = []
            onnx_registry = None
            if aten_conversion_changes is not None:
                onnx_registry = OnnxRegistry()
                for fct, name in aten_conversion_changes:
                    reg_name = name
                    onnx_registry.register_op(
                        function=fct,
                        namespace="aten",
                        op_name=reg_name,
                        overload="default",
                    )
                    assert isinstance(
                        reg_name, str
                    ), f"Wrong type {type(reg_name)}, name={name}, fct={fct}"
                    names.append(f"torch.ops.aten.{reg_name}.default")
                    if verbose:
                        print(f"[make_aot_ort] register {names[-1]!r}")

            # from onnxruntime.training.ortmodule.torch_cpp_extensions.cuda
            # import load_aten_op_executor_cpp_extension
            from onnxruntime.training.ortmodule.torch_cpp_extensions import (
                aten_op_executor,
            )
            from onnxruntime.capi import _pybind_state as _C

            _C.register_aten_op_executor(
                str(aten_op_executor.is_tensor_argument_address()),
                str(aten_op_executor.execute_aten_operator_address()),
            )

            ort_session_options = onnxruntime.SessionOptions()
            # ort_session_options.log_severity_level = 1

            if onnx_registry is None:
                export_options = ExportOptions(dynamic_shapes=dynamic)
            else:
                if verbose:
                    print(f"[make_aot_ort] enable {onnx_registry!r}")
                export_options = ExportOptions(
                    dynamic_shapes=dynamic, onnx_registry=onnx_registry
                )

            if rewrite:

                from onnxscript.optimizer import optimize
                from onnxscript.rewriter import rewrite
                from onnx.inliner import inline_local_functions

                def optimize_model_proto_oxs(model_proto):
                    first_model_proto = model_proto
                    model_proto = optimize(
                        model_proto,
                        num_iterations=2,
                        onnx_shape_inference=False,
                    )
                    model_proto = rewrite(model_proto)
                    model_proto = inline_local_functions(model_proto)
                    del first_model_proto.graph.node[:]
                    del first_model_proto.functions[:]
                    first_model_proto.graph.node.extend(model_proto.graph.node)
                    first_model_proto.functions.extend(model_proto.functions)
                    modify_onnx(first_model_proto)
                    return first_model_proto

                if verbose:
                    print("[make_aot_ort] enable rewriting")

                options = OrtBackendOptions(
                    export_options=export_options,
                    ort_session_options=ort_session_options,
                    pre_ort_model_transforms=[optimize_model_proto_oxs],
                )
            else:
                options = OrtBackendOptions(
                    export_options=export_options,
                    ort_session_options=ort_session_options,
                )

            ort_backend = OrtBackend(options=options)

            if names:
                for n in names:
                    ort_backend._supported_ops._support_dict[n] = None

            return ort_backend, ort_backend

        aten_conversion_changes = [
            (
                scaled_dot_product_efficient_attention,
                "_scaled_dot_product_efficient_attention",
            ),
            (
                scaled_dot_product_attention_backward,
                "_scaled_dot_product_efficient_attention_backward",
            ),
        ]

        local_aot_ort, _ = make_aot_ort(
            dynamic=False,  # True,
            rewrite=True,
            aten_conversion_changes=aten_conversion_changes,
            verbose=1,
        )

        config = LlamaConfig(
            hidden_size=16,
            num_hidden_layers=1,
            vocab_size=1024,
            intermediate_size=16,
            max_position_embeddings=1024,
            num_attention_heads=2,
        )
        config._attn_implementation = "sdpa"

        model = LlamaModel(config).to("cuda")

        batch, seq, vocab_size = 2, 1024, 1024

        input_ids = ids_tensor([batch, seq], vocab_size).to("cuda")
        # input_mask = torch.tril(torch.ones(batch, seq, dtype=torch.float32))

        model(input_ids)  # , input_mask)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            optimized_mod = torch.compile(model, backend=local_aot_ort, fullgraph=True)
            with dump_onnx(
                "dort-llama-sdpa-ort", folder="dump_sdpa_oxs_llama", clean=True
            ):
                output = optimized_mod(input_ids)  # , input_mask)
                output[0].sum().backward()

        names = [_ for _ in os.listdir("dump_sdpa_oxs_llama") if _.endswith(".onnx")]
        print("------------------------------------------")
        print(f"exported model: {names}")
        for name in names:
            print()
            print("NODES in {name!r}")
            onx = onnx.load(os.path.join("dump_sdpa_oxs_llama", name))
            for i, node in enumerate(onx.graph.node):
                print(
                    f"{i+1}/{len(onx.graph.node)}: "
                    f"{node.op_type} {node.input} -> {node.output}"
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
