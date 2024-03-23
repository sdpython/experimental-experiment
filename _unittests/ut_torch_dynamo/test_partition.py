import unittest
import torch
from torch._dynamo.backends.common import aot_autograd
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    skipif_ci_windows,
    skipif_ci_apple,
)
from experimental_experiment.torch_dynamo import (
    onnx_debug_backend,
    get_decomposition_table,
)
from experimental_experiment.torch_dynamo.partition import (
    get_partition_fn,
    backend_partition_compile,
    CustomOperatorSupport,
)


class TestPartition(ExtTestCase):

    @skipif_ci_apple("no onnxruntime-training")
    @skipif_ci_windows("no torch dynamo")
    def test_nopartition_debug(self):

        class MLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.Sequential(
                    torch.nn.Linear(10, 32),
                    torch.nn.Sigmoid(),
                    torch.nn.Linear(32, 1),
                )

            def forward(self, x):
                return self.layers(x)

        x = torch.randn(3, 10, dtype=torch.float32)

        mlp = MLP()
        expected = mlp(x)

        backend_debug = lambda *args, **kwargs: onnx_debug_backend(  # noqa: E731
            *args,
            backend="ref",
            target_opset=18,
            verbose=0,
        )

        aot_compiler = aot_autograd(
            fw_compiler=backend_debug,
            decompositions=get_decomposition_table(),
        )

        compiled_model = torch.compile(
            mlp, backend=aot_compiler, dynamic=False, fullgraph=True
        )

        got = compiled_model(x)
        self.assertEqualArray(expected, got, atol=1e-5)

    @skipif_ci_apple("no onnxruntime-training")
    @skipif_ci_windows("no torch dynamo")
    @unittest.skipIf(True, reason="not implemented yet")
    def test_1_partition_sigmoid_debug(self):

        class MLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.Sequential(
                    torch.nn.Linear(10, 32),
                    torch.nn.Sigmoid(),
                    torch.nn.Linear(32, 1),
                )

            def forward(self, x):
                return self.layers(x)

        x = torch.randn(3, 10, dtype=torch.float32)

        mlp = MLP()
        expected = mlp(x)

        def backend_debug(*args, **kwargs):
            return onnx_debug_backend(
                *args, backend="ref", target_opset=18, verbose=1, **kwargs
            )

        support = CustomOperatorSupport(unsupport_dict={}, verbose=2)

        aot_compiler = aot_autograd(
            fw_compiler=lambda *args, **kwargs: backend_partition_compile(
                *args,
                support=support,
                backend_function=backend_debug,
                verbose=1,
                use_aot_autograd=True,
                decompositions=get_decomposition_table(),
                partition_fn=get_partition_fn(),
                **kwargs,
            ),
            decompositions=get_decomposition_table(),
            partition_fn=get_partition_fn(),
        )

        compiled_model = torch.compile(
            mlp, backend=aot_compiler, dynamic=False, fullgraph=True
        )

        got = compiled_model(x)
        self.assertEqualArray(expected, got, atol=1e-5)

    @skipif_ci_apple("no onnxruntime-training")
    @skipif_ci_windows("no torch dynamo")
    @unittest.skipIf(True, reason="not implemented yet")
    def test_partition_sigmoid_debug(self):

        class MLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.Sequential(
                    torch.nn.Linear(10, 32),
                    torch.nn.Sigmoid(),
                    torch.nn.Linear(32, 1),
                )

            def forward(self, x):
                return self.layers(x)

        x = torch.randn(3, 10, dtype=torch.float32)

        mlp = MLP()
        expected = mlp(x)

        def backend_debug(*args, **kwargs):
            return onnx_debug_backend(
                *args, backend="ref", target_opset=18, verbose=1, **kwargs
            )

        support = CustomOperatorSupport(
            unsupport_dict={"torch.ops.aten.sigmoid.default"}, verbose=2
        )

        aot_compiler = aot_autograd(
            fw_compiler=lambda *args, **kwargs: backend_partition_compile(
                *args,
                support=support,
                backend_function=backend_debug,
                verbose=1,
                use_aot_autograd=True,
                decompositions=get_decomposition_table(),
                partition_fn=get_partition_fn(),
                **kwargs,
            ),
            decompositions=get_decomposition_table(),
            partition_fn=get_partition_fn(),
        )

        compiled_model = torch.compile(
            mlp, backend=aot_compiler, dynamic=False, fullgraph=True
        )

        got = compiled_model(x)
        self.assertEqualArray(expected, got, atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
