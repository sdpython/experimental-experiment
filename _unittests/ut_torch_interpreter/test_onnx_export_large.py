import os
import unittest
import warnings
from typing import Optional
from onnx.reference import ReferenceEvaluator
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    skipif_ci_windows,
)
from experimental_experiment.xbuilder import OptimizationOptions
from experimental_experiment.torch_interpreter import to_onnx, ExportOptions
from experimental_experiment.helpers import pretty_onnx


class TestOnnxExportLarge(ExtTestCase):

    def return_module_cls_relu(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import torch
            from torch import nn

        class MyModel(nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()
                self.conv1 = nn.Conv2d(1, 128, 5)

            def forward(self, x):
                return torch.relu(self.conv1(x))

        input_tensor = torch.rand((1, 1, 128, 128), dtype=torch.float32)
        return MyModel(), input_tensor

    def export_utils(
        self,
        prefix,
        model,
        *args,
        remove_unused=False,
        constant_folding=True,
        verbose=0,
        rename_input=True,
        expected_weights=None,
    ):
        if not os.path.exists(prefix):
            os.mkdir(prefix)
        names = []
        name = os.path.join(prefix, "large.onnx")
        if os.path.exists(name):
            os.remove(name)
        large_onx = to_onnx(
            model,
            tuple(args),
            input_names=["input"] if rename_input else None,
            options=OptimizationOptions(
                remove_unused=remove_unused,
                constant_folding=constant_folding,
                verbose=verbose,
                patterns=None,
            ),
            verbose=verbose,
            large_model=True,
        )
        if expected_weights is not None:
            assert len(large_onx.model_proto.graph.initializer) == expected_weights, (
                f"The model has {len(large_onx.model_proto.graph.initializer)} "
                f"initiliazers, expecting {expected_weights}, inputs are "
                f"{[_.name for _ in large_onx.model_proto.graph.input]}."
            )
        large_onx.save(name)
        names.append(name)
        return names

    def check_model_ort(self, name):
        from onnxruntime import InferenceSession

        if isinstance(name, str):
            try:
                InferenceSession(name, providers=["CPUExecutionProvider"])
            except Exception as e:
                import onnx

                raise AssertionError(  # noqa: B904
                    f"onnxruntime cannot load the model "
                    f"due to {e}\n{pretty_onnx(onnx.load(name))}"
                )
            return
        try:
            InferenceSession(name.SerializeToString(), providers=["CPUExecutionProvider"])
        except Exception as e:
            raise AssertionError(  # noqa: B904
                f"onnxruntime cannot load the model due to {e}\n{pretty_onnx(name)}"
            )

    @skipif_ci_windows("torch dynamo not supported on windows")
    @ignore_warnings((UserWarning, DeprecationWarning))
    def test_export_as_large_model(self):
        model, input_tensor = self.return_module_cls_relu()
        names = self.export_utils("test_export_as_large_model", model, input_tensor)
        x = input_tensor.numpy()
        results = []
        for name in names:
            ref = ReferenceEvaluator(name)
            results.append(ref.run(None, {"input": x})[0])
            self.check_model_ort(name)
        if len(results) == 2:
            self.assertEqualArray(results[0], results[1])

    def test_issue_onnx_7465_knn(self):
        # https://github.com/onnx/onnx/issues/7465
        try:
            import torch_cluster as _  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("torch-cluster is not installed")

        import torch
        import torch.nn as nn
        from torch_geometric.nn import GravNetConv, global_mean_pool
        from torch_geometric.nn.aggr import MeanAggregation

        def make_patches():
            def knn(
                x: torch.Tensor,
                y: torch.Tensor,
                k: int,
                batch_x: Optional[torch.Tensor] = None,
                batch_y: Optional[torch.Tensor] = None,
                cosine: bool = False,
                num_workers: int = 1,
                batch_size: Optional[int] = None,
            ) -> torch.Tensor:
                if x.numel() == 0 or y.numel() == 0:
                    return torch.empty(2, 0, dtype=torch.long, device=x.device)

                x = x.view(-1, 1) if x.dim() == 1 else x
                y = y.view(-1, 1) if y.dim() == 1 else y
                x, y = x.contiguous(), y.contiguous()

                if batch_size is None:
                    batch_size = 1
                    if batch_x is not None:
                        # assert x.size(0) == batch_x.numel()
                        # PATCHED
                        batch_size = batch_x.max().item() + 1
                    if batch_y is not None:
                        # assert y.size(0) == batch_y.numel()
                        # PATCHED
                        batch_size = torch.sym_max(batch_size, batch_y.max().item() + 1)
                assert batch_size > 0

                ptr_x: Optional[torch.Tensor] = None
                ptr_y: Optional[torch.Tensor] = None
                # if batch_size > 1:
                # PATCHED
                if True:
                    assert batch_x is not None
                    assert batch_y is not None
                    arange = torch.arange(batch_size + 1, device=x.device)
                    ptr_x = torch.bucketize(arange, batch_x)
                    ptr_y = torch.bucketize(arange, batch_y)

                assert k == 8
                assert not cosine
                assert num_workers == 1
                res = torch.ops.torch_cluster.knn_py(x, y, ptr_x, ptr_y)
                return res

            @torch.library.custom_op("torch_cluster::knn_py", mutates_args={})
            def knn_py(
                x: torch.Tensor, y: torch.Tensor, ptr_x: torch.Tensor, ptr_y: torch.Tensor
            ) -> torch.Tensor:
                return torch.ops.torch_cluster.knn(x, y, ptr_x, ptr_y, 8, False, 1)

            @knn_py.register_fake
            def knn_py_shape(x, y, ptr_x, ptr_y):
                k = 8
                return torch.empty((2, y.shape[1] * k), dtype=torch.int64, device=x.device)

        in_ch = 16
        hidden = 32
        out_ch = 5

        class GNNet(nn.Module):
            def __init__(self):
                super().__init__()
                P = 16
                self.conv = GravNetConv(
                    in_ch, hidden, space_dimensions=4, propagate_dimensions=P, k=8
                )
                self.act = nn.ReLU()
                self.head = nn.Linear(hidden, out_ch)
                self.conv.aggr_module = MeanAggregation()
                self.conv.lin_out2 = nn.Linear(P, self.conv.lin_out2.out_channels, bias=True).to(
                    self.conv.lin_out2.weight
                )

            def forward(self, x, batch):
                h = self.conv(x, batch)
                h = self.act(h)
                return self.head(global_mean_pool(h, batch))

        total_nodes = 128
        batch_size = 4

        x = torch.randn(total_nodes, in_ch)

        nodes_per_graph = total_nodes // batch_size
        remainder = total_nodes % batch_size

        sizes = torch.full((batch_size,), nodes_per_graph, dtype=torch.long)
        sizes[:remainder] += 1

        graph_ids = torch.arange(batch_size, dtype=torch.long)
        batch = torch.repeat_interleave(graph_ids, sizes)
        model = GNNet().eval()
        expected = model(x, batch)
        self.assertNotEmpty(expected)

        torch.export.export(model, (x, batch))
        onx = to_onnx(
            model,
            (x, batch),
            dynamic_shapes={"x": {0: "num_nodes"}, "batch": {0: "num_nodes"}},
            export_options=ExportOptions(decomposition_table="all"),
        )
        self.dump_onnx("test_issue_onnx_7465.onnx", onx)


if __name__ == "__main__":
    unittest.main(verbosity=2)
