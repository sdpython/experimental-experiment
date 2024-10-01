import unittest
import onnx.helper as oh
from onnx import TensorProto
from experimental_experiment.ext_test_case import ExtTestCase, hide_stdout
from experimental_experiment.model_run import model_run


class TestCommandLines(ExtTestCase):

    @hide_stdout()
    def test_model_run(self):
        proto = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Sigmoid", ["Y"], ["sy"]),
                    oh.make_node("Mul", ["Y", "sy"], ["ysy"]),
                    oh.make_node("Mul", ["X", "ysy"], ["final"]),
                ],
                "nd",
                [
                    oh.make_tensor_value_info("X", TensorProto.FLOAT, ["a", 5, 6]),
                    oh.make_tensor_value_info("Y", TensorProto.FLOAT, ["a", 5, 6]),
                ],
                [oh.make_tensor_value_info("final", TensorProto.FLOAT, ["a", "b", "c"])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )

        stats = model_run(proto, verbose=1, validate=proto)
        self.assertIn("time_latency", stats)


if __name__ == "__main__":
    unittest.main(verbosity=2)
