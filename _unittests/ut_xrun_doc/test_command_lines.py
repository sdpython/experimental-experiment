import unittest
from contextlib import redirect_stdout
from io import StringIO
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment._command_lines_parser import (
    get_main_parser,
    get_parser_lighten,
    get_parser_print,
    get_parser_unlighten,
)


class TestCommandLines(ExtTestCase):
    def test_main_parser(self):
        st = StringIO()
        with redirect_stdout(st):
            get_main_parser().print_help()
        text = st.getvalue()
        self.assertIn("lighten", text)

    def test_parser_lighten(self):
        st = StringIO()
        with redirect_stdout(st):
            get_parser_lighten().print_help()
        text = st.getvalue()
        self.assertIn("model", text)

    def test_parser_unlighten(self):
        st = StringIO()
        with redirect_stdout(st):
            get_parser_unlighten().print_help()
        text = st.getvalue()
        self.assertIn("model", text)

    def test_parser_print(self):
        st = StringIO()
        with redirect_stdout(st):
            get_parser_print().print_help()
        text = st.getvalue()
        self.assertIn("pretty", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
