import copy
import os
import unittest
import onnx.helper as oh
from onnx import TensorProto
import torch
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    skipif_ci_windows,
    skipif_ci_apple,
    has_cuda,
)


class TestBackend(ExtTestCase):

    @ignore_warnings(DeprecationWarning)
    @skipif_ci_windows("onnxruntime-training not available")
    @skipif_ci_apple("onnxruntime-training not available")
    def test_onnx_custom_backend_dump(self):
        import onnxruntime
        from experimental_experiment.torch_dynamo.fast_backend import OrtBackend
        from experimental_experiment.torch_helper.dump_helper import assert_all_close

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "Y"], ["Z"]),
                    oh.make_node("Cast", ["X"], ["xi"], to=TensorProto.INT64),
                    oh.make_node("Add", ["S", "xi"], ["Z2"]),
                    oh.make_node("Identity", ["input_dim_0"], ["output_dim_0"]),
                    oh.make_node("Identity", ["input_dim_1"], ["output_dim_1"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TensorProto.FLOAT, [None, None]),
                    oh.make_tensor_value_info("Y", TensorProto.FLOAT, [None, None]),
                    oh.make_tensor_value_info("S", TensorProto.INT64, [None, None]),
                    oh.make_tensor_value_info("input_dim_0", TensorProto.INT64, [1]),
                    oh.make_tensor_value_info("input_dim_1", TensorProto.INT64, [1]),
                ],
                [
                    oh.make_tensor_value_info("Z", TensorProto.FLOAT, [None, None]),
                    oh.make_tensor_value_info("Z2", TensorProto.INT64, [None, None]),
                    oh.make_tensor_value_info("output_dim_0", TensorProto.INT64, [1]),
                    oh.make_tensor_value_info("output_dim_1", TensorProto.INT64, [1]),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        sess = onnxruntime.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        bck = OrtBackend(sess, onnx_model=model)
        inputs = [
            torch.arange(9, dtype=torch.float32).reshape((-1, 3)),
            torch.arange(9, dtype=torch.float32).reshape((-1, 3)),
            torch.arange(9, dtype=torch.int64).reshape((-1, 3)),
            2,
            1024,
        ]
        res = bck(*inputs)
        assert_all_close(inputs[0] + inputs[1], res[0])
        self.assertEqual(res[2], 2)
        self.assertEqual(res[3], 1024)
        res = bck(*inputs)
        assert_all_close(inputs[0] + inputs[1], res[0])
        self.assertEqual(res[2], 2)
        self.assertEqual(res[3], 1024)
        bck.dump_for_debug("temp_data", *inputs)
        self.assertExists("temp_data")
        self.assertExists("temp_data/model.onnx")
        self.assertExists("temp_data/test_case_0/input_0.pb")
        self.assertExists("temp_data/test_case_0/input_1.pb")

        new_bck, new_inputs = OrtBackend.replay_dumped_data("temp_data")
        self.assertEqual(len(inputs), len(new_inputs))
        assert_all_close(tuple(inputs), tuple(new_inputs))
        res2 = new_bck(*new_inputs)
        self.assertEqual(len(res), len(res2))
        assert_all_close(tuple(res), tuple(res2))

    def test_debug_data(self):
        from experimental_experiment.torch_dynamo.fast_backend import OrtBackend

        tttype = {
            TensorProto.FLOAT: torch.float32,
            TensorProto.INT64: torch.int64,
            TensorProto.FLOAT16: torch.float16,
        }
        if not os.path.exists("debug_data"):
            raise unittest.SkipTest("debug_data does not exist")
        for providers in [
            ["CPUExecutionProvider"],
            (
                [("CUDAExecutionProvider", {}), ("CPUExecutionProvider", {})]
                if has_cuda()
                else None
            ),
        ]:
            if providers is None:
                continue
            new_bck, new_inputs = OrtBackend.replay_dumped_data(
                "debug_data", providers=providers
            )
            for i in range(len(new_inputs)):
                v = new_inputs[i]
                dt = new_bck.onnx_model.graph.input[i]
                dti = dt.type.tensor_type.elem_type
                if isinstance(v, torch.Tensor):
                    self.assertEqual(
                        tttype[dti],
                        v.dtype,
                        msg=f"wrong type, i={i}, name={dt.name!r}",
                    )
                else:
                    self.assertEqual(
                        dti,
                        TensorProto.INT64,
                        msg=f"wrong type, i={i}, name={dt.name!r}",
                    )
            for i in range(0, 3):
                with self.subTest(providers=providers, i=i):
                    res = new_bck(*new_inputs)
                    for r, name in zip(res, new_bck.output_names):
                        if "_dim_" in name:
                            assert isinstance(
                                r, int
                            ), f"unexpected type {type(r)} for name={name!r}"
                            assert r != 0, f"unexpected value {r} for name={name!r}"

    @skipif_ci_windows("no torch dynamo")
    def test_transforms_debug(self):
        from experimental_experiment.torch_dynamo import onnx_debug_backend

        stored = []

        def store_model(m):
            stored.append(m)
            return m

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

        compiled_model = torch.compile(
            copy.deepcopy(mlp),
            backend=lambda *args, **kwargs: onnx_debug_backend(
                *args, target_opset=18, pre_ort_model_transforms=store_model, **kwargs
            ),
            dynamic=False,
            fullgraph=True,
        )

        got = compiled_model(x)
        self.assertEqualArray(expected, got, atol=1e-5)
        self.assertNotEmpty(stored)

    @skipif_ci_apple("no onnxruntime-training")
    @skipif_ci_windows("no torch dynamo")
    def test_transforms_custom(self):
        from experimental_experiment.torch_dynamo import onnx_custom_backend

        stored = []

        def store_model(m):
            stored.append(m)
            return m

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

        compiled_model = torch.compile(
            copy.deepcopy(mlp),
            backend=lambda *args, **kwargs: onnx_custom_backend(
                *args, target_opset=18, pre_ort_model_transforms=store_model, **kwargs
            ),
            dynamic=False,
            fullgraph=True,
        )

        got = compiled_model(x)
        self.assertEqualArray(expected, got, atol=1e-5)
        self.assertNotEmpty(stored)

    @skipif_ci_windows("no torch dynamo")
    def test_backend_dynger(self):
        from experimental_experiment.torch_dynamo import dynger_backend

        stored = []

        def store_model(m):
            stored.append(m)
            return m

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

        compiled_model = torch.compile(
            copy.deepcopy(mlp),
            backend=lambda *args, **kwargs: dynger_backend(*args, **kwargs),
            dynamic=False,
            fullgraph=True,
        )

        got = compiled_model(x)
        self.assertEqualArray(expected, got, atol=1e-5)

    @skipif_ci_apple("no onnxruntime-training")
    @skipif_ci_windows("no torch dynamo")
    def test_ort_graph_no_optimization(self):
        from experimental_experiment.torch_dynamo import onnx_custom_backend

        stored = []

        def store_model(m):
            stored.append(m)
            return m

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

        compiled_model = torch.compile(
            copy.deepcopy(mlp),
            backend=lambda *args, **kwargs: onnx_custom_backend(
                *args,
                target_opset=18,
                pre_ort_model_transforms=store_model,
                ort_optimization_level="ORT_DISABLE_ALL",
                **kwargs,
            ),
            dynamic=False,
            fullgraph=True,
        )

        got = compiled_model(x)
        self.assertEqualArray(expected, got, atol=1e-5)
        self.assertNotEmpty(stored)


if __name__ == "__main__":
    unittest.main(verbosity=2)
