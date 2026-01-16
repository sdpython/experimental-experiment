import unittest
from onnx import save as onnx_save
from onnx.inliner import inline_local_functions
from onnx.checker import check_model
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    skipif_ci_windows,
    requires_torch,
    ignore_warnings,
)
from experimental_experiment.xbuilder import FunctionOptions
from experimental_experiment.torch_interpreter import to_onnx, ExportOptions
from experimental_experiment.xbuilder import OptimizationOptions
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.torch_test_helper import dummy_llm


class TestOnnxExportSubModules(ExtTestCase):

    @skipif_ci_windows("not available on windows")
    @requires_torch("2.6", "owning module is None before that")
    @ignore_warnings(FutureWarning)
    def test_submodule_local_functions_simple(self):
        import torch

        class SubNeuron2(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.linear = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                z = self.linear(x)
                return torch.sigmoid(z)

        class Neuron2(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.neuron = SubNeuron2(n_dims, n_targets)

            def forward(self, x):
                z = self.neuron(x)
                return torch.relu(z)

        model = Neuron2()
        inputs = (torch.randn(1, 5),)
        expected = model(*inputs)
        feeds = {"x": inputs[0].numpy()}

        onx = to_onnx(
            model,
            inputs,
            export_modules_as_functions=True,
            optimize=False,
            verbose=0,
            inline=False,
        )
        check_model(onx)
        self.assertEqual(len(onx.functions), 2)
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)

    @skipif_ci_windows("not available on windows")
    @requires_torch("2.6", "owning module is None before that")
    @ignore_warnings(FutureWarning)
    def test_submodule_local_functions_double(self):
        import torch

        class SubNeuron2(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.linear1 = torch.nn.Linear(n_dims, n_targets)
                self.linear2 = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                z1 = self.linear1(x)
                z2 = self.linear2(x)
                return torch.sigmoid(z1) + torch.sigmoid(z2)

        class Neuron2(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.neuron = SubNeuron2(n_dims, n_targets)

            def forward(self, x):
                z = self.neuron(x)
                return torch.relu(z)

        model = Neuron2()
        inputs = (torch.randn(1, 5),)
        expected = model(*inputs)
        feeds = {"x": inputs[0].numpy()}

        onx = to_onnx(
            model,
            inputs,
            export_modules_as_functions=True,
            optimize=False,
            verbose=0,
            function_options=FunctionOptions(merge_allowed=True, external_threshold=0),
            inline=False,
        )
        check_model(onx)
        self.assertEqual(len(onx.functions), 2)
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)

    @skipif_ci_windows("not available on windows")
    @requires_torch("2.6", "owning module is None before that")
    @ignore_warnings(FutureWarning)
    def test_submodule_local_functions_two_outputs(self):
        import torch

        class SubNeuron2Outputs(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()

            def forward(self, x):
                return (
                    torch.sigmoid(x),
                    torch.sigmoid(x * x),
                )

        class Neuron2(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.neuron = SubNeuron2Outputs(n_dims, n_targets)

            def forward(self, x):
                z, z1 = self.neuron(x)
                return torch.relu(z) + z1

        model = Neuron2()
        inputs = (torch.randn(1, 5),)
        expected = model(*inputs)
        feeds = {"x": inputs[0].numpy()}

        onx = to_onnx(
            model,
            inputs,
            export_modules_as_functions=True,
            optimize=False,
            verbose=0,
            inline=False,
        )
        check_model(onx)
        self.assertEqual(len(onx.functions), 1)
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)

    @skipif_ci_windows("bug")
    @ignore_warnings(FutureWarning)
    def test_dummy_llm_flat_strict_true(self):
        model, inputs = dummy_llm()
        onx = to_onnx(
            model, inputs, optimize=False, verbose=0, export_options=ExportOptions(strict=True)
        )
        names = [i.name for i in onx.graph.initializer]
        self.assertNotIn("p_decoder_feed_forward_linear_1_weight", names)
        self.check_ort(onx)

    def test_dummy_llm_flat_strict_false(self):
        model, inputs = dummy_llm()
        onx = to_onnx(
            model,
            inputs,
            optimize=False,
            verbose=0,
            export_options=ExportOptions(strict=False),
        )
        names = [i.name for i in onx.graph.initializer]
        self.assertNotIn("p_decoder_feed_forward_linear_1_weight", names)
        self.check_ort(onx)

    @requires_torch("2.11", "flacky")
    @ignore_warnings(FutureWarning)
    def test_dummy_llm_strict_true(self):
        model, inputs = dummy_llm()
        onx = to_onnx(
            model,
            inputs,
            export_modules_as_functions=True,
            optimize=False,
            verbose=0,
            export_options=ExportOptions(strict=True),
            inline=False,
        )
        self.dump_onnx("test_dummy_llm_strict_true.onnx", onx)
        node_names = [n.op_type for n in onx.graph.node]
        self.assertEqual(node_names, ["<locals>.Embedding", "<locals>.DecoderLayer", "Identity"])
        node_names = [n.op_type for n in onx.functions[1].node]
        self.assertEqual(node_names, ["Embedding", "Embedding", "Add", "Identity"])
        p_names = set(name for name, _ in model.named_parameters())
        init_names = set(i.name for i in onx.graph.initializer if "mask" not in i.name)
        removed = {
            "decoder.attention.linear.bias",
            "decoder.norm_1.bias",
            "decoder.attention.attention.0.mask",
            "decoder.feed_forward.linear_2.bias",
            "decoder.norm_2.bias",
            "decoder.norm_1.weight",
            "decoder.norm_2.weight",
            "decoder.attention.attention.1.mask",
        }
        self.assertEqual(init_names, p_names - removed)
        check_model(onx)
        self.check_ort(onx)
        for name, param in model.named_parameters():
            if name.endswith(".bias"):
                # probably a bug, fix it when this feature is needed
                continue
            assert name not in removed or name.endswith(".mask") or param.min() == param.max(), (
                f"unexpected removed parameter for {name!r} and "
                f"{self.string_type(param, with_shape=True, with_min_max=True)}"
            )

        onx2 = to_onnx(model, inputs, optimize=False, verbose=0)
        self.dump_onnx("test_dummy_llm_strict_true2.onnx", onx)
        init_names2 = set(i.name for i in onx2.graph.initializer if "mask" not in i.name)
        self.assertEqual(init_names2 & init_names, init_names)
        self.check_ort(onx2)

    @requires_torch("2.6", "owning_module is None")
    @ignore_warnings(FutureWarning)
    def test_dummy_llm_opts(self):
        model, inputs = dummy_llm()
        onx = to_onnx(
            model,
            inputs,
            optimize=True,
            verbose=0,
            export_options=ExportOptions(strict=False),
            inline=False,
            options=OptimizationOptions(patterns="default+onnxruntime", constant_folding=False),
            export_modules_as_functions=True,
        )
        self.dump_onnx("test_dummy_llm_opts.onnx", onx)
        node_names = [n.op_type for n in onx.graph.node]
        self.assertEqual(node_names, ["<locals>.Embedding", "<locals>.DecoderLayer"])
        node_names = [n.op_type for n in onx.functions[1].node]
        self.assertEqual(node_names, ["Embedding", "Embedding", "Add"])
        p_names = set(name for name, _ in model.named_parameters())
        init_names = set(i.name for i in onx.graph.initializer if "mask" not in i.name)
        self.assertEqual(len(p_names & init_names), 12)
        check_model(onx)
        self.check_ort(onx)

        onx2 = to_onnx(
            model,
            inputs,
            optimize=True,
            verbose=0,
            options=OptimizationOptions(patterns="default+onnxruntime", constant_folding=False),
        )
        init_names2 = set(i.name for i in onx2.graph.initializer if "mask" not in i.name)
        self.dump_onnx("test_dummy_llm_opts.2.onnx", onx2)
        self.check_ort(onx2)
        self.assertEqual(init_names2 & init_names, init_names)

    @requires_torch("2.6", "owning_module is None")
    @ignore_warnings(FutureWarning)
    def test_dummy_llm_strict_pieces_true(self):
        for cls_name in ["DecoderLayer", "AttentionBlock", "MultiAttentionBlock"]:
            with self.subTest(cls_name=cls_name):
                model, inputs = dummy_llm(cls_name)
                onx2 = to_onnx(model, inputs, optimize=False, verbose=0)
                onnx_save(onx2, f"test_dummy_llm_strict_pieces_true_{cls_name}.onnx")
                self.check_ort(onx2)
                onx2 = to_onnx(model, inputs, optimize=True, verbose=0)
                self.check_ort(onx2)
                onx = to_onnx(
                    model,
                    inputs,
                    export_modules_as_functions=True,
                    optimize=False,
                    verbose=0,
                )
                onnx_save(onx, f"test_dummy_llm_strict_pieces_true_{cls_name}.module.onnx")
                check_model(onx)
                inlined = inline_local_functions(onx)
                onnx_save(
                    inlined,
                    f"test_dummy_llm_strict_pieces_true_{cls_name}.module.inlined.onnx",
                )
                check_model(inlined)
                self.check_ort(inlined)
                self.check_ort(onx)

    @requires_torch("2.6", "owning_module is None")
    @ignore_warnings(FutureWarning)
    def test_dummy_llm_strict_false(self):
        model, inputs = dummy_llm()
        onx = to_onnx(
            model,
            inputs,
            export_modules_as_functions=True,
            optimize=False,
            verbose=0,
            export_options=ExportOptions(strict=False),
            inline=False,
        )
        node_names = [n.op_type for n in onx.graph.node]
        self.assertEqual(node_names, ["<locals>.Embedding", "<locals>.DecoderLayer", "Identity"])
        node_names = [n.op_type for n in onx.functions[1].node]
        self.assertEqual(node_names, ["Embedding", "Embedding", "Add", "Identity"])
        p_names = set(name for name, _ in model.named_parameters())
        init_names = set(i.name for i in onx.graph.initializer if "mask" not in i.name)
        self.assertEqual(len(p_names & init_names), 12)
        check_model(onx)
        self.check_ort(onx)

        onx2 = to_onnx(model, inputs, optimize=False, verbose=0)
        init_names2 = set(i.name for i in onx2.graph.initializer if "mask" not in i.name)
        self.assertEqual(init_names2 & init_names, init_names)
        self.check_ort(onx2)

    @ignore_warnings(FutureWarning)
    def test_dummy_llm_opts_inline(self):
        model, inputs = dummy_llm()
        onx = to_onnx(
            model,
            inputs,
            optimize=True,
            verbose=0,
            export_options=ExportOptions(strict=False),
            inline=True,
            options=OptimizationOptions(patterns="default+onnxruntime", constant_folding=False),
            export_modules_as_functions=True,
        )
        self.dump_onnx("test_dummy_llm_opts.onnx", onx)

    @skipif_ci_windows("not available on windows")
    @requires_torch("2.6", "owning module is None before that")
    @ignore_warnings(FutureWarning)
    def test_submodule_local_functions_more_depth_static(self):
        import torch

        class Level1(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.linear1 = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                z1 = self.linear1(x)
                ones = torch.ones(z1.shape, dtype=z1.dtype, device=z1.device)
                ones[0, 0] = 0
                return torch.sigmoid(z1 + ones)

        class Level2(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.sublevela = Level1(n_dims, n_targets)
                self.linear2 = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                y1 = self.linear2(x)
                y2 = self.sublevela(x)
                ones = torch.ones(y2.shape, dtype=y2.dtype, device=y2.device)
                ones[0, 0] = 0
                return torch.sigmoid(y1 + ones) + y2

        class Level3(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.sublevelb = Level2(n_dims, n_targets)
                self.linear3 = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                w1 = self.linear3(x)
                w2 = self.sublevelb(x)
                ones = torch.ones(w2.shape, dtype=w2.dtype, device=w2.device)
                ones[0, 0] = 0
                return torch.sigmoid(w1 + ones) + w2

        model = Level3()
        inputs = (torch.randn(1, 5),)
        expected = model(*inputs)
        feeds = {"x": inputs[0].numpy()}

        filename = self.get_dump_file("test_submodule_local_functions_more_depth_static.onnx")
        onx = to_onnx(
            model,
            inputs,
            export_modules_as_functions=True,
            optimize=False,
            verbose=0,
            function_options=FunctionOptions(merge_allowed=True, external_threshold=0),
            inline=False,
            filename=filename,
        )
        check_model(onx)
        self.assertEqual(len(onx.functions), 3)
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)

    @skipif_ci_windows("not available on windows")
    @requires_torch("2.6", "owning module is None before that")
    @ignore_warnings(FutureWarning)
    def test_submodule_local_functions_more_depth_dynamic1(self):
        import torch

        class Level1(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.linear1 = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                z1 = self.linear1(x)
                ones = torch.ones(z1.shape, dtype=z1.dtype, device=z1.device)
                ones[0, 0] = 0
                return torch.sigmoid(z1 + ones)

        class Level2(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.sublevela = Level1(n_dims, n_targets)
                self.linear2 = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                y1 = self.linear2(x)
                y2 = self.sublevela(x)
                ones = torch.ones(y2.shape, dtype=y2.dtype, device=y2.device)
                ones[0, 0] = 0
                return torch.sigmoid(y1 + ones) + y2

        model = Level2()
        inputs = (torch.randn(2, 5),)
        expected = model(*inputs)
        feeds = {"x": inputs[0].numpy()}

        filename = self.get_dump_file("test_submodule_local_functions_more_depth_dynamic1.onnx")
        onx = to_onnx(
            model,
            inputs,
            export_modules_as_functions=True,
            optimize=False,
            verbose=0,
            function_options=FunctionOptions(merge_allowed=True, external_threshold=0),
            inline=False,
            filename=filename,
            dynamic_shapes=({0: "batch"},),
        )
        check_model(onx)
        self.assertEqual(len(onx.functions), 2)
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)

    @skipif_ci_windows("not available on windows")
    @requires_torch("2.6", "owning module is None before that")
    @ignore_warnings(FutureWarning)
    def test_submodule_local_functions_more_depth_dynamic2(self):
        import torch

        class Level1(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.linear1 = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                z1 = self.linear1(x)
                ones = torch.ones(z1.shape, dtype=z1.dtype, device=z1.device)
                ones[0, 0] = 0
                return torch.sigmoid(z1 + ones)

        class Level2(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.sublevela = Level1(n_dims, n_targets)
                self.linear2 = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                y1 = self.linear2(x)
                y2 = self.sublevela(x)
                ones = torch.ones(y2.shape, dtype=y2.dtype, device=y2.device)
                ones[0, 0] = 0
                return torch.sigmoid(y1 + ones) + y2

        class Level3(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.sublevelb = Level2(n_dims, n_targets)
                self.linear3 = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                w1 = self.linear3(x)
                w2 = self.sublevelb(x)
                ones = torch.ones(w2.shape, dtype=w2.dtype, device=w2.device)
                ones[0, 0] = 0
                return torch.sigmoid(w1 + ones) + w2

        model = Level3()
        inputs = (torch.randn(2, 5),)
        expected = model(*inputs)
        feeds = {"x": inputs[0].numpy()}

        filename = self.get_dump_file("test_submodule_local_functions_more_depth_dynamic2.onnx")
        onx = to_onnx(
            model,
            inputs,
            export_modules_as_functions=True,
            optimize=False,
            verbose=0,
            function_options=FunctionOptions(merge_allowed=True, external_threshold=0),
            inline=False,
            filename=filename,
            dynamic_shapes=({0: "batch"},),
        )
        check_model(onx)
        self.assertEqual(len(onx.functions), 3)
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)

    @skipif_ci_windows("not available on windows")
    @requires_torch("2.6", "owning module is None before that")
    @ignore_warnings(FutureWarning)
    def test_submodule_local_functions_more_depth_dynamic2_preserve2(self):
        import torch

        class Level1(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.linear1 = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                z1 = self.linear1(x)
                ones = torch.ones(z1.shape, dtype=z1.dtype, device=z1.device)
                ones[0, 0] = 0
                return torch.sigmoid(z1 + ones)

        class Level2(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.sublevela = Level1(n_dims, n_targets)
                self.linear2 = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                y1 = self.linear2(x)
                y2 = self.sublevela(x)
                ones = torch.ones(y2.shape, dtype=y2.dtype, device=y2.device)
                ones[0, 0] = 0
                return torch.sigmoid(y1 + ones) + y2

        class Level3(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.sublevelb = Level2(n_dims, n_targets)
                self.linear3 = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                w1 = self.linear3(x)
                w2 = self.sublevelb(x)
                ones = torch.ones(w2.shape, dtype=w2.dtype, device=w2.device)
                ones[0, 0] = 0
                return torch.sigmoid(w1 + ones) + w2

        model = Level3()
        inputs = (torch.randn(2, 5),)
        expected = model(*inputs)
        feeds = {"x": inputs[0].numpy()}

        filename = self.get_dump_file(
            "test_submodule_local_functions_more_depth_dynamic2_preserve2.onnx"
        )
        onx = to_onnx(
            model,
            inputs,
            export_modules_as_functions={Level2},
            optimize=False,
            verbose=0,
            function_options=FunctionOptions(merge_allowed=True, external_threshold=0),
            inline=False,
            filename=filename,
            dynamic_shapes=({0: "batch"},),
        )
        check_model(onx)
        self.assertEqual(len(onx.functions), 1)
        self.assertEqual(["<locals>.Level2"], [f.name for f in onx.functions])
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)

    @skipif_ci_windows("not available on windows")
    @requires_torch("2.6", "owning module is None before that")
    @ignore_warnings(FutureWarning)
    def test_submodule_local_functions_more_depth_dynamic2_preserve1(self):
        import torch

        class Level1(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.linear1 = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                z1 = self.linear1(x)
                ones = torch.ones(z1.shape, dtype=z1.dtype, device=z1.device)
                ones[0, 0] = 0
                return torch.sigmoid(z1 + ones)

        class Level2(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.sublevela = Level1(n_dims, n_targets)
                self.linear2 = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                y1 = self.linear2(x)
                y2 = self.sublevela(x)
                ones = torch.ones(y2.shape, dtype=y2.dtype, device=y2.device)
                ones[0, 0] = 0
                return torch.sigmoid(y1 + ones) + y2

        class Level3(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.sublevelb = Level2(n_dims, n_targets)
                self.linear3 = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                w1 = self.linear3(x)
                w2 = self.sublevelb(x)
                ones = torch.ones(w2.shape, dtype=w2.dtype, device=w2.device)
                ones[0, 0] = 0
                return torch.sigmoid(w1 + ones) + w2

        model = Level3()
        inputs = (torch.randn(2, 5),)
        expected = model(*inputs)
        feeds = {"x": inputs[0].numpy()}

        filename = self.get_dump_file(
            "test_submodule_local_functions_more_depth_dynamic2_preserve1.onnx"
        )
        onx = to_onnx(
            model,
            inputs,
            export_modules_as_functions={Level1},
            optimize=False,
            verbose=0,
            function_options=FunctionOptions(merge_allowed=True, external_threshold=0),
            inline=False,
            filename=filename,
            dynamic_shapes=({0: "batch"},),
        )
        check_model(onx)
        self.assertEqual(len(onx.functions), 1)
        self.assertEqual(["<locals>.Level1"], [f.name for f in onx.functions])
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)

    @skipif_ci_windows("not available on windows")
    @requires_torch("2.6", "owning module is None before that")
    @ignore_warnings(FutureWarning)
    def test_submodule_local_functions_more_depth_dynamic1_2io(self):
        import torch

        class Level1(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.linear1 = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                z1 = self.linear1(x)
                ones = torch.ones(z1.shape, dtype=z1.dtype, device=z1.device)
                ones[0, 0] = 0
                return z1, torch.sigmoid(z1 + ones)

        class Level2(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.sublevela = Level1(n_dims, n_targets)
                self.linear2 = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                y1 = self.linear2(x)
                y2, y3 = self.sublevela(x)
                ones = torch.ones(y2.shape, dtype=y2.dtype, device=y2.device)
                ones[0, 0] = 0
                return torch.sigmoid(y1 + y2 + ones) + y3

        model = Level2()
        inputs = (torch.randn(2, 5),)
        expected = model(*inputs)
        feeds = {"x": inputs[0].numpy()}

        filename = self.get_dump_file(
            "test_submodule_local_functions_more_depth_dynamic1_2io.onnx"
        )
        onx = to_onnx(
            model,
            inputs,
            export_modules_as_functions=True,
            optimize=False,
            verbose=0,
            function_options=FunctionOptions(merge_allowed=True, external_threshold=0),
            inline=False,
            filename=filename,
            dynamic_shapes=({0: "batch"},),
        )
        check_model(onx)
        self.assertEqual(len(onx.functions), 2)
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)

    @skipif_ci_windows("not available on windows")
    @requires_torch("2.6", "owning module is None before that")
    @ignore_warnings(FutureWarning)
    def test_submodule_local_functions_more_depth_dynamic2_preserve2_2io1(self):
        import torch

        class Level1(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.linear1 = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                z1 = self.linear1(x)
                ones = torch.ones(z1.shape, dtype=z1.dtype, device=z1.device)
                ones[0, 0] = 0
                return z1, torch.sigmoid(z1 + ones)

        class Level2(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.sublevela = Level1(n_dims, n_targets)
                self.linear2 = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                y1 = self.linear2(x)
                y2, y3 = self.sublevela(x)
                ones = torch.ones(y2.shape, dtype=y2.dtype, device=y2.device)
                ones[0, 0] = 0
                return torch.sigmoid(y1 + ones + y2) + y3

        class Level3(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.sublevelb = Level2(n_dims, n_targets)
                self.linear3 = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                w1 = self.linear3(x)
                w2 = self.sublevelb(x)
                ones = torch.ones(w2.shape, dtype=w2.dtype, device=w2.device)
                ones[0, 0] = 0
                return torch.sigmoid(w1 + ones) + w2

        model = Level3()
        inputs = (torch.randn(2, 5),)
        expected = model(*inputs)
        feeds = {"x": inputs[0].numpy()}

        filename = self.get_dump_file(
            "test_submodule_local_functions_more_depth_dynamic2_preserve2_2io1.onnx"
        )
        onx = to_onnx(
            model,
            inputs,
            export_modules_as_functions={Level2},
            optimize=False,
            verbose=0,
            function_options=FunctionOptions(merge_allowed=True, external_threshold=0),
            inline=False,
            filename=filename,
            dynamic_shapes=({0: "batch"},),
        )
        check_model(onx)
        self.assertEqual(len(onx.functions), 1)
        self.assertEqual(["<locals>.Level2"], [f.name for f in onx.functions])
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)

    @skipif_ci_windows("not available on windows")
    @requires_torch("2.6", "owning module is None before that")
    @ignore_warnings(FutureWarning)
    def test_submodule_local_functions_shapes(self):
        import torch

        class Level1(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.linear1 = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                z1 = self.linear1(x)
                ones = torch.ones(z1.shape, dtype=z1.dtype, device=z1.device)
                ones[0, 0] = 0
                return z1.shape[0], torch.sigmoid(z1 + ones)

        class Level2(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.sublevela = Level1(n_dims, n_targets)
                self.linear2 = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                y1 = self.linear2(x)
                batch_dim, y3 = self.sublevela(x)
                ones = torch.ones((batch_dim, y3.shape[1]), dtype=y3.dtype, device=y3.device)
                ones[0, 0] = 0
                return torch.sigmoid(y1 + ones) + y3

        class Level3(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.sublevelb = Level2(n_dims, n_targets)
                self.linear3 = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                w1 = self.linear3(x)
                w2 = self.sublevelb(x)
                ones = torch.ones(w2.shape, dtype=w2.dtype, device=w2.device)
                ones[0, 0] = 0
                return torch.sigmoid(w1 + ones) + w2

        model = Level3()
        inputs = (torch.randn(2, 5),)
        expected = model(*inputs)
        feeds = {"x": inputs[0].numpy()}

        filename = self.get_dump_file("test_submodule_local_functions_shapes.onnx")
        onx = to_onnx(
            model,
            inputs,
            export_modules_as_functions={Level1},
            optimize=True,
            verbose=0,
            function_options=FunctionOptions(merge_allowed=True, external_threshold=0),
            inline=False,
            filename=filename,
            dynamic_shapes=({0: "batch"},),
        )
        check_model(onx)
        self.assertEqual(len(onx.functions), 1)
        self.assertEqual(["<locals>.Level1"], [f.name for f in onx.functions])
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
