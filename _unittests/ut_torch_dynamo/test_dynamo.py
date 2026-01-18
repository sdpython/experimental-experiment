import unittest
import textwrap
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.torch_dynamo import pprint_storage


class TestDynamo(ExtTestCase):
    def test_pprint_storage(self):
        obs = {"j": 3, "hh": 5}
        o = pprint_storage(obs)
        self.assertEqual(o, "{'j': 3, 'hh': 5}")

        obs = [5, 6]
        o = pprint_storage(obs)
        self.assertEqual(o, "[5, 6]")

        obs = (6, 7)
        o = pprint_storage(obs)
        self.assertEqual(o, "(6, 7)")

        obs = {"j": 3, "hh": 5, "ll": [77]}
        o = pprint_storage(obs).strip(" \n")
        expected = textwrap.dedent("""
        {
          'j': 3,
          'hh': 5,
          'll': [77],
        }
        """).strip(" \n")
        self.assertEqual(expected, o)

        obs = {"j": 3, "hh": 5, "ll": [(77, 55)]}
        o = pprint_storage(obs).strip(" \n")
        expected = textwrap.dedent("""
        {
          'j': 3,
          'hh': 5,
          'll': [
            (77, 55),
          ],
        }
        """).strip(" \n")
        self.assertEqual(expected, o)

    def test_pprint_storage_long(self):
        obs = {"j": 3, "hh": 5, "ll": [4, [77, 55, 0, 1, 2, 3, 4, 5, 6, 8, 9, 10]]}
        o = pprint_storage(obs).strip(" \n")
        expected = textwrap.dedent("""
        {
          'j': 3,
          'hh': 5,
          'll': [
            4,
            [
              77,
              55,
              0,
              1,
              2,
              3,
              4,
              5,
              6,
              8,
              9,
              10,
            ],
          ],
        }
        """).strip(" \n")
        self.assertEqual(expected, o)


if __name__ == "__main__":
    unittest.main(verbosity=2)
