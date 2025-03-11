import textwrap
import unittest
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.torch_interpreter.control_flows import refactor_if_else_functions


class TestControlFlows(ExtTestCase):
    def test_rewrite_code_if(self):
        original_code = textwrap.dedent(
            """
        def compute():
            x = 5
            y = 10
            if x > 3:
                y = y * 2
                x = x + 1
            else:
                y = y + 3
                x = x - 1
            return x, y
        """
        ).strip(" \n")

        expected = (
            textwrap.dedent(
                """
        def compute():
            x = 5
            y = 10
            if x > 3:
                x, y = branch_func1(x, y)
            else:
                x, y = branch_func2(x, y)
            return (x, y)

        def branch_func1(x, y):
            y = y * 2
            x = x + 1
            return (x, y)

        def branch_func2(x, y):
            y = y + 3
            x = x - 1
            return (x, y)
        """
            ).strip(" \n"),
            textwrap.dedent(
                """
        def compute():
            x = 5
            y = 10
            if x > 3:
                y, x = branch_func1(y, x)
            else:
                y, x = branch_func2(y, x)
            return (x, y)

        def branch_func1(y, x):
            y = y * 2
            x = x + 1
            return (y, x)

        def branch_func2(y, x):
            y = y + 3
            x = x - 1
            return (y, x)
        """
            ).strip(" \n"),
        )

        # Extract if-else branches
        new_code = refactor_if_else_functions(original_code)
        self.maxDiff = None
        self.assertIn(new_code, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
