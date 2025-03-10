import unittest
from typing import Optional
import numpy as np
from onnx import (
    TensorProto,
    helper as oh,
    numpy_helper as onh,
)
from onnx_array_api.translate_api.make_helper import make_node_extended
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.ext_test_case import ExtTestCase, requires_cuda
from experimental_experiment.xbuilder.graph_builder import (
    GraphBuilder,
    OptimizationOptions,
)

TFLOAT = TensorProto.FLOAT
TINT64 = TensorProto.INT64


class TestGraphPatternOptimizationOrtAttention(ExtTestCase):
    def _range(self, *shape, bias: Optional[float] = None):
        n = np.prod(shape)
        x = np.cos(np.arange(n)).astype(np.float32)
        if bias:
            x = x + bias
        return x.reshape(tuple(shape)).astype(np.float32)

    def _get_model_attention_1(self):
        opset_imports = [
            oh.make_opsetid("pkg.onnxscript.torch_lib.common", 1),
            oh.make_opsetid("", 18),
            oh.make_opsetid("pkg.onnxscript.torch_lib", 1),
            oh.make_opsetid("pkg.torch.__subgraph__", 1),
            oh.make_opsetid("com.microsoft", 1),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        initializers.append(
            onh.from_array(
                self._range(1024, 1024),
                name="encoder.encoders.0.self_attn.linear_q.weight",
            )
        )
        initializers.append(
            onh.from_array(
                self._range(1024),
                name="encoder.encoders.0.self_attn.linear_q.bias",
            )
        )
        initializers.append(
            onh.from_array(
                self._range(1024, 1024),
                name="encoder.encoders.0.self_attn.linear_k.weight",
            )
        )
        initializers.append(
            onh.from_array(
                self._range(1024),
                name="encoder.encoders.0.self_attn.linear_k.bias",
            )
        )
        initializers.append(
            onh.from_array(
                self._range(1024, 1024),
                name="encoder.encoders.0.self_attn.linear_v.weight",
            )
        )
        initializers.append(
            onh.from_array(
                self._range(1024),
                name="encoder.encoders.0.self_attn.linear_v.bias",
            )
        )
        initializers.append(onh.from_array(np.array(1, dtype=np.int64), name="dim_0_7"))
        initializers.append(onh.from_array(np.array(0, dtype=np.int64), name="val_10"))
        initializers.append(
            onh.from_array(np.array(-np.inf, dtype=np.float32), name="val_124")
        )
        initializers.append(onh.from_array(np.array(0.0, dtype=np.float32), name="val_126"))
        inputs.append(
            oh.make_tensor_value_info(
                "layer_norm_1", TensorProto.FLOAT, shape=("s0", "(s1-1)//8+1", 1024)
            )
        )
        inputs.append(
            oh.make_tensor_value_info(
                "expand_1",
                TensorProto.BOOL,
                shape=("s0", "(s1-1)//8+1", "(s1-1)//8+1"),
            )
        )
        inputs.append(
            oh.make_tensor_value_info(
                "unsqueeze_9", TensorProto.FLOAT, shape=(1, 16, "(s1-1)//8+1", "(s1-1)//8+1")
            )
        )
        inputs.append(oh.make_tensor_value_info("val_104", TensorProto.INT64, shape=(4,)))
        inputs.append(oh.make_tensor_value_info("val_112", TensorProto.INT64, shape=(4,)))
        inputs.append(oh.make_tensor_value_info("val_120", TensorProto.INT64, shape=(4,)))
        inputs.append(oh.make_tensor_value_info("val_132", TensorProto.INT64, shape=(3,)))
        nodes.append(make_node_extended("Unsqueeze", ["expand_1", "dim_0_7"], ["unsqueeze_6"]))
        nodes.append(
            make_node_extended("Cast", ["unsqueeze_6"], ["convert_element_type_default"], to=7)
        )
        nodes.append(
            make_node_extended("Equal", ["convert_element_type_default", "val_10"], ["eq_87"])
        )
        nodes.append(
            make_node_extended(
                "MatMul",
                ["layer_norm_1", "encoder.encoders.0.self_attn.linear_q.weight"],
                ["val_97"],
            )
        )
        nodes.append(
            make_node_extended(
                "Add", ["val_97", "encoder.encoders.0.self_attn.linear_q.bias"], ["linear_3"]
            )
        )
        nodes.append(
            make_node_extended("Reshape", ["linear_3", "val_104"], ["view"], allowzero=0)
        )
        nodes.append(
            make_node_extended(
                "MatMul",
                ["layer_norm_1", "encoder.encoders.0.self_attn.linear_k.weight"],
                ["val_107"],
            )
        )
        nodes.append(
            make_node_extended(
                "Add", ["val_107", "encoder.encoders.0.self_attn.linear_k.bias"], ["linear_4"]
            )
        )
        nodes.append(
            make_node_extended("Reshape", ["linear_4", "val_112"], ["view_1"], allowzero=0)
        )
        nodes.append(
            make_node_extended(
                "MatMul",
                ["layer_norm_1", "encoder.encoders.0.self_attn.linear_v.weight"],
                ["val_115"],
            )
        )
        nodes.append(
            make_node_extended(
                "Add", ["val_115", "encoder.encoders.0.self_attn.linear_v.bias"], ["linear_5"]
            )
        )
        nodes.append(
            make_node_extended("Reshape", ["linear_5", "val_120"], ["view_2"], allowzero=0)
        )
        nodes.append(
            make_node_extended("Transpose", ["view"], ["transpose_1"], perm=[0, 2, 1, 3])
        )
        nodes.append(
            make_node_extended(
                "Transpose",
                ["view_1"],
                ["TransposeFusedMatMulBPattern--transpose_4"],
                perm=[0, 2, 1, 3],
            )
        )
        nodes.append(
            make_node_extended(
                "FusedMatMul",
                ["transpose_1", "TransposeFusedMatMulBPattern--transpose_4"],
                ["matmul"],
                domain="com.microsoft",
                transB=1,
                alpha=0.125,
            )
        )
        nodes.append(
            make_node_extended("Transpose", ["view_2"], ["transpose_3"], perm=[0, 2, 1, 3])
        )
        nodes.append(make_node_extended("Add", ["matmul", "unsqueeze_9"], ["add_322"]))
        nodes.append(
            make_node_extended("Where", ["eq_87", "val_124", "add_322"], ["masked_fill_2"])
        )
        nodes.append(make_node_extended("Softmax", ["masked_fill_2"], ["softmax"], axis=-1))
        nodes.append(
            make_node_extended("Where", ["eq_87", "val_126", "softmax"], ["masked_fill_3"])
        )
        nodes.append(
            make_node_extended("MatMul", ["masked_fill_3", "transpose_3"], ["matmul_1"])
        )
        nodes.append(
            make_node_extended("Transpose", ["matmul_1"], ["transpose_5"], perm=[0, 2, 1, 3])
        )
        nodes.append(
            make_node_extended("Reshape", ["transpose_5", "val_132"], ["view_3"], allowzero=0)
        )
        outputs.append(
            oh.make_tensor_value_info(
                "view_3", TensorProto.FLOAT, shape=("s0", "(s1-1)//8+1", 1024)
            )
        )
        graph = oh.make_graph(
            nodes,
            "attention_pattern",
            inputs,
            outputs,
            initializers,
            sparse_initializer=sparse_initializers,
        )
        model = oh.make_model(
            graph, functions=functions, opset_imports=opset_imports, ir_version=10
        )
        return model

    @requires_cuda("Attention 4D not available on CPU")
    def test_attention_pattern_1_4d_cuda(self):
        model = self._get_model_attention_1()
        self.dump_onnx("test_attention_pattern_1.noopt.onnx", model)
        feeds = {
            "layer_norm_1": self._range(2, 8, 1024),  # s0,(s1-1)//8+1,1024
            "expand_1": np.random.randint(0, 2, size=(2, 8, 8))
            > 0,  # s0,CeilToInt(IntTrueDiv(s1, 8)),CeilToInt(IntTrueDiv(s1, 8))
            "unsqueeze_9": self._range(1, 16, 8, 8),  # 1,16,(s1-1)//8+1,(s1-1)//8+1
            "val_104": np.array([2, 8, 16, 64], dtype=np.int64),  # s0,(s1-1)//8+1,16,6
            "val_112": np.array([2, 8, 16, 64], dtype=np.int64),  # s0,(s1-1)//8+1,16,6
            "val_120": np.array([2, 8, 16, 64], dtype=np.int64),  # s0,(s1-1)//8+1,16,6
            "val_132": np.array(
                [2, 8, 1024], dtype=np.int64
            ),  # s0,CeilToInt(IntTrueDiv(s1, 8)),1024
        }
        # cap = ExtendedReferenceEvaluator(model, verbose=10)
        # cap.run(None, feeds)
        from onnxruntime import InferenceSession

        ref = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["Attention"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertIn("Attention", [n.op_type for n in opt_onx.graph.node])

        opt_ref = InferenceSession(
            opt_onx.SerializeToString(),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        got = opt_ref.run(None, feeds)
        self.assertEqualArray(expected[0].ravel(), got[0].ravel(), atol=0.1)
        self.assertEqualArray(expected[0], got[0], atol=0.1)

    def test_attention_pattern_1_4d_cpu(self):
        model = self._get_model_attention_1()
        self.dump_onnx("test_attention_pattern_1.noopt.onnx", model)
        feeds = {
            "layer_norm_1": self._range(2, 8, 1024),  # s0,(s1-1)//8+1,1024
            "expand_1": np.random.randint(0, 2, size=(2, 8, 8))
            > 0,  # s0,CeilToInt(IntTrueDiv(s1, 8)),CeilToInt(IntTrueDiv(s1, 8))
            "unsqueeze_9": self._range(1, 16, 8, 8),  # 1,16,(s1-1)//8+1,(s1-1)//8+1
            "val_104": np.array([2, 8, 16, 64], dtype=np.int64),  # s0,(s1-1)//8+1,16,6
            "val_112": np.array([2, 8, 16, 64], dtype=np.int64),  # s0,(s1-1)//8+1,16,6
            "val_120": np.array([2, 8, 16, 64], dtype=np.int64),  # s0,(s1-1)//8+1,16,6
            "val_132": np.array(
                [2, 8, 1024], dtype=np.int64
            ),  # s0,CeilToInt(IntTrueDiv(s1, 8)),1024
        }
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(patterns=["Attention"], verbose=0),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertIn("Attention", [n.op_type for n in opt_onx.graph.node])

        opt_ref = ExtendedReferenceEvaluator(opt_onx)
        got = opt_ref.run(None, feeds)
        self.assertEqualArray(expected[0].ravel(), got[0].ravel())
        self.assertEqualArray(expected[0], got[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
