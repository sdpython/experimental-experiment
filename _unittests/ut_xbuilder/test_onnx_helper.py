import unittest
import onnx.helper as oh
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.xbuilder._onnx_helper import same_function_proto


class TestOnnxHelper(ExtTestCase):
    def test_same_function_proto(self):

        f1 = oh.make_function(
            "custom",
            "LinearRegression",
            ["x", "a", "b"],
            ["y"],
            [
                oh.make_node("MatMul", ["x", "a"], ["xa"]),
                oh.make_node("Add", ["xa", "b"], ["y"]),
            ],
            [oh.make_opsetid("", 14)],
            [],
        )
        self.assertEqualTrue(same_function_proto(f1, f1, verbose=1))
        f2 = oh.make_function(
            "custom",
            "LinearRegression",
            ["x_", "a_", "b_"],
            ["y_"],
            [
                oh.make_node("MatMul", ["x_", "a_"], ["xa_"]),
                oh.make_node("Add", ["xa_", "b_"], ["y_"]),
            ],
            [oh.make_opsetid("", 14)],
            [],
        )
        self.assertEqualTrue(same_function_proto(f1, f2, verbose=1))
        f3 = oh.make_function(
            "custom",
            "LinearRegression",
            ["x_", "a_", "b_"],
            ["y_"],
            [
                oh.make_node("MatMul", ["x_", "a_"], ["xa_"]),
                oh.make_node("Add", ["xb_", "a_"], ["y_"]),
            ],
            [oh.make_opsetid("", 14)],
            [],
        )
        self.assertEqual(
            same_function_proto(f1, f3, verbose=1),
            "different input names at node 1, ['xa', 'b'], ['xb_', 'a_'] != ['xa_', 'b_']",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
