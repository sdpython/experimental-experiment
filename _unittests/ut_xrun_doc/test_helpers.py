import unittest
import numpy as np
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.helpers import string_type, string_sig


class TestHelpers(ExtTestCase):
    def test_string_type(self):
        a = np.array([1])
        obj = {"a": a, "b": [5.6], "c": (1,)}
        s = string_type(obj)
        self.assertEqual(s, "dict(a:A1,b:[float],c:(int,))")

    def test_string_sig_f(self):

        def f(a, b=3, c=4, e=5):
            pass

        ssig = string_sig(f, {"a": 1, "c": 8, "b": 3})
        self.assertEqual(ssig, "f(a=1, c=8)")

    def test_string_sig_cls(self):

        class A:
            def __init__(self, a, b=3, c=4, e=5):
                self.a, self.b, self.c, self.e = a, b, c, e

        ssig = string_sig(A(1, c=8))
        self.assertEqual(ssig, "A(a=1, c=8)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
