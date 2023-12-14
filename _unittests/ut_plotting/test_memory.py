import unittest
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.plotting.data import memory_peak_plot_data
from experimental_experiment.plotting.memory import memory_peak_plot


class TestPlottingMemory(ExtTestCase):
    def test_memory_peak_plot(self):
        data = memory_peak_plot_data()
        ax = memory_peak_plot(
            data,
            suptitle="nice",
            bars=[55, 110],
            key=("export", "aot", "compute"),
            figsize=(18 * 2, 7 * 2),
        )
        self.assertNotEmpty(ax)
        ax[0, 0].get_figure().savefig("check.png")


if __name__ == "__main__":
    unittest.main(verbosity=2)
