import unittest
from experimental_experiment.ext_test_case import ExtTestCase, requires_cuda


class TestCustomOpsDispatch(ExtTestCase):
    @requires_cuda()
    def test_llama_sdpa_model_efficient(self):
        # see https://pytorch.org/tutorials/beginner/onnx/onnx_registry_tutorial.html
        # python -m pip install torch_ort
        # python -m torch_ort configure
        import contextlib
        import os
        import warnings
        from typing import Any, Dict, List, Optional
        import numpy as np
        import onnx
        import torch
        import torch.onnx
        from torch._dynamo.backends.common import aot_autograd
        from experimental_experiment.torch_interpreter import Dispatcher
        from experimental_experiment.torch_dynamo import onnx_custom_backend
        from experimental_experiment.torch_dynamo import get_decomposition_table

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

        #
        # onnx part
        #

        def onnx_scaled_dot_product_efficient_attention(
            g: "GraphBuilder",  # noqa: F821
            sts: Dict[str, Any],
            outputs: List[str],
            query,
            key,
            value,
            attn_bias,
            compute_log_sumexp: bool,
            dropout_p: float,
            is_causal: bool,
            scale: float = 1.0,
            **kwargs,
        ):
            assert (
                len(outputs) == 4
            ), f"Unexpected number of outputs {outputs}{g.get_debug_msg()}"
            assert len(kwargs) == 0, (
                f"Unexpected kwargs {kwargs} in "
                f"onnx_scaled_dot_product_efficient_attention{g.get_debug_msg()}"
            )
            # itype = g.get_type(value)
            # dtype = tensor_dtype_to_np_dtype(itype)
            t_compute_log_sumexp = g.make_initializer(
                "", np.array(compute_log_sumexp, dtype=np.bool_)
            )
            t_dropout_p = g.make_initializer("", np.array(dropout_p, dtype=np.float32))
            t_is_causal = g.make_initializer("", np.array(is_causal, dtype=np.bool_))
            t_scale = g.make_initializer("", np.array(scale or 1.0, dtype=np.float32))
            output, log_sumexp, philox_seed, philox_offset = g.make_node(
                "ATen",
                [
                    query,
                    key,
                    value,
                    attn_bias or "",
                    t_compute_log_sumexp,
                    t_dropout_p,
                    t_is_causal,
                    t_scale,
                ],
                outputs=outputs,
                operator="_scaled_dot_product_efficient_attention",
                domain="org.pytorch.aten",
                name="scaled_dot_product_efficient_attention",
            )
            g.add_domain("org.pytorch.aten")
            return output, log_sumexp, philox_seed, philox_offset

        def onnx_scaled_dot_product_attention_backward(
            g: "GraphBuilder",  # noqa: F821
            sts: Dict[str, Any],
            outputs: List[str],
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
            is_causal: bool,
            scale: float = 1.0,
            **kwargs,
        ):
            assert (
                len(outputs) == 4
            ), f"Unexpected number of outputs {outputs}{g.get_debug_msg()}"
            assert len(kwargs) == 0, (
                f"Unexpected kwargs {kwargs} in "
                f"onnx_scaled_dot_product_attention_backward{g.get_debug_msg()}"
            )
            t_scale = g.make_initializer("", np.array(scale or 1.0, dtype=np.float32))
            t_dropout_p = g.make_initializer("", np.array(dropout_p, dtype=np.float32))
            t_is_causal = g.make_initializer("", np.array(is_causal, dtype=np.bool_))
            t_grad_input_mask = g.make_initializer(
                "", np.array(grad_input_mask, dtype=np.int64)
            )
            # onnxruntime fails with type inference failed
            # Let's add some Cast even if not needed.
            dt = g.get_type(grad)
            helper = ",".join(map(str, [dt, dt, dt, dt]))
            node_name = f"scaled_dot_product_attention_backward[{helper}]"
            grad_query, grad_key, grad_value, grad_attn_bias = g.make_node(
                "ATen",
                [
                    grad,
                    query,
                    key,
                    value,
                    attn_bias or "",
                    output,
                    logsumexp,
                    philox_seed,
                    philox_offset,
                    t_dropout_p,
                    t_grad_input_mask,
                    t_is_causal,
                    t_scale,
                ],
                outputs=outputs,
                operator="_scaled_dot_product_efficient_attention_backward",
                domain="org.pytorch.aten",
                name=node_name,
            )
            g.add_domain("org.pytorch.aten")
            return grad_query, grad_key, grad_value, grad_attn_bias

        dispatcher = Dispatcher(
            {
                "_scaled_dot_product_efficient_attention_default": onnx_scaled_dot_product_efficient_attention,  # noqa: E501
                "_scaled_dot_product_efficient_attention_backward_default": onnx_scaled_dot_product_attention_backward,  # noqa: E501
            }
        )

        #
        # model part
        #

        def ids_tensor(shape, vocab_size):
            total_dims = 1
            for dim in shape:
                total_dims *= dim

            values = []
            for _ in range(total_dims):
                values.append(np.random.randint(0, vocab_size - 1))

            return torch.tensor(data=values, dtype=torch.long).view(shape).contiguous()

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

        #
        # onnxruntime and dynamo part
        #

        from onnxruntime.training.ortmodule.torch_cpp_extensions import (
            aten_op_executor,
        )
        from onnxruntime.capi import _pybind_state as _C

        _C.register_aten_op_executor(
            str(aten_op_executor.is_tensor_argument_address()),
            str(aten_op_executor.execute_aten_operator_address()),
        )

        aot_compiler = aot_autograd(
            fw_compiler=lambda *args, **kwargs: onnx_custom_backend(
                *args,
                target_opset=18,
                dispatcher=dispatcher,
                verbose=0,
                optimize=True,
                **kwargs,
            ),
            decompositions=get_decomposition_table(),
        )

        #
        # test starts
        #

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            optimized_mod = torch.compile(model, backend=aot_compiler, fullgraph=True)
            with dump_onnx(
                "dort-llama-sdpa-custom-no", folder="dump_sdpa_dis_llama", clean=True
            ):
                output = optimized_mod(input_ids)  # , input_mask)
                output[0].sum().backward()

        names = [_ for _ in os.listdir("dump_sdpa_dis_llama") if _.endswith(".onnx")]
        print("------------------------------------------")
        print(f"exported model: {names}")
        for name in names:
            print()
            print("NODES in {name!r}")
            onx = onnx.load(os.path.join("dump_sdpa_dis_llama", name))
            for i, node in enumerate(onx.graph.node):
                print(
                    f"{i+1}/{len(onx.graph.node)}: "
                    f"{node.op_type} {node.input} -> {node.output}"
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
