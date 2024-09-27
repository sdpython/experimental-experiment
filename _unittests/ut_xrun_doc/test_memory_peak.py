import os
import time
import unittest
import numpy as np
from experimental_experiment.ext_test_case import ExtTestCase, skipif_ci_apple
from experimental_experiment.memory_peak import get_memory_rss, start_spying_on
import torch


class TestMemoryPeak(ExtTestCase):
    @skipif_ci_apple("stuck")
    def test_memory(self):
        mem = get_memory_rss(os.getpid())
        self.assertIsInstance(mem, int)

    @skipif_ci_apple("stuck")
    def test_spy(self):
        p = start_spying_on(cuda=False)
        n_elements = 0
        for _i in range(10):
            time.sleep(0.005)
            value = np.empty(2**23, dtype=np.int64)
            value += 1
            n_elements = max(value.shape[0], n_elements)
        time.sleep(0.02)
        pres = p.stop()
        self.assertIsInstance(pres, dict)
        self.assertLessEqual(pres["cpu"].end, pres["cpu"].max_peak)
        self.assertLessEqual(pres["cpu"].begin, pres["cpu"].max_peak)
        self.assertGreater(pres["cpu"].delta_peak, 0)
        self.assertGreaterOrEqual(pres["cpu"].delta_peak, pres["cpu"].delta_end)
        self.assertGreaterOrEqual(pres["cpu"].delta_peak, pres["cpu"].delta_avg)
        self.assertGreater(pres["cpu"].delta_end, 0)
        self.assertGreater(pres["cpu"].delta_avg, 0)
        self.assertGreater(pres["cpu"].delta_peak, n_elements * 8 * 0.5)
        self.assertIsInstance(pres["cpu"].to_dict(), dict)

    @skipif_ci_apple("stuck")
    @unittest.skipIf(not torch.cuda.is_available(), reason="CUDA not here")
    def test_spy_cuda(self):
        p = start_spying_on(cuda=True)
        n_elements = 0
        for _i in range(10):
            time.sleep(0.005)
            value = torch.empty(2**23, dtype=torch.int64, device="cuda")
            value += 1
            n_elements = max(value.shape[0], n_elements)
        time.sleep(0.02)
        pres = p.stop()
        self.assertIsInstance(pres, dict)
        self.assertIn("gpus", pres)
        gpu = pres["gpus"][0]
        self.assertLessEqual(gpu.end, gpu.max_peak)
        self.assertLessEqual(gpu.begin, gpu.max_peak)
        self.assertGreater(gpu.delta_peak, 0)
        self.assertGreaterOrEqual(gpu.delta_peak, gpu.delta_end)
        self.assertGreaterOrEqual(gpu.delta_peak, gpu.delta_avg)
        self.assertGreater(gpu.delta_end, 0)
        self.assertGreater(gpu.delta_avg, 0)
        self.assertGreater(gpu.delta_peak, n_elements * 8 * 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
