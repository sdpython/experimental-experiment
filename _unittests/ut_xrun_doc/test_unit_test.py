import os
import unittest
import pandas
import experimental_experiment
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    statistics_on_file,
    statistics_on_folder,
)


class TestUnitTest(ExtTestCase):
    def test_statistics_on_file(self):
        stat = statistics_on_file(__file__)
        self.assertEqual(stat["ext"], ".py")
        self.assertGreater(stat["lines"], 8)
        self.assertGreater(stat["chars"], stat["lines"])

    def test_statistics_on_folder(self):
        stat = statistics_on_folder(
            os.path.join(os.path.dirname(__file__), ".."), aggregation=1
        )
        self.assertGreater(len(stat), 1)

        df = pandas.DataFrame(stat)
        gr = df.drop("name", axis=1).groupby(["dir", "ext"]).sum()
        self.assertEqual(len(gr.columns), 2)

    def test_statistics_on_folders(self):
        stat = statistics_on_folder(
            [
                os.path.join(os.path.dirname(experimental_experiment.__file__)),
                os.path.join(os.path.dirname(experimental_experiment.__file__), "..", "_doc"),
                os.path.join(
                    os.path.dirname(experimental_experiment.__file__),
                    "..",
                    "_unittests",
                ),
            ],
            aggregation=2,
        )
        self.assertGreater(len(stat), 1)

        df = pandas.DataFrame(stat)
        gr = df.drop("name", axis=1).groupby(["ext", "dir"]).sum().reset_index()
        gr = gr[gr["dir"] != "_doc/auto_examples"]
        total = (
            gr[gr["dir"].str.contains("experimental_experiment/")]
            .drop(["ext", "dir"], axis=1)
            .sum(axis=0)
        )
        self.assertEqual(len(gr.columns), 4)
        self.assertEqual(total.shape, (2,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
