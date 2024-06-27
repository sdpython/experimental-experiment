import os
import time
import unittest
import sys
import numpy as np
from experimental_experiment.ext_test_case import ExtTestCase, skipif_ci_apple
from experimental_experiment.memory_peak import get_memory_rss, start_spying_on
import torch


class TestMemoryPeak(ExtTestCase):
    @skipif_ci_apple("stuck")
    @unittest.skipIf(sys.version_info >= (3, 12, 0), reason="too long")
    def test_memory(self):
        mem = get_memory_rss(os.getpid())
        self.assertIsInstance(mem, int)

    @skipif_ci_apple("stuck")
    @unittest.skipIf(sys.version_info >= (3, 12, 0), reason="stuck")
    def test_spy(self):
        p = start_spying_on()
        res = []
        for i in range(0, 10):
            time.sleep(0.005)
            res.append(np.empty(i * 1000000))
        del res
        time.sleep(0.02)
        pres = p.stop()
        self.assertIsInstance(pres, dict)
        self.assertLessEqual(pres["cpu"].end, pres["cpu"].max_peak)
        self.assertLessEqual(pres["cpu"].begin, pres["cpu"].max_peak)
        self.assertIsInstance(pres["cpu"].to_dict(), dict)

    @skipif_ci_apple("stuck")
    @unittest.skipIf(not torch.cuda.is_available(), reason="CUDA not here")
    @unittest.skipIf(sys.version_info >= (3, 12, 0), reason="too long")
    def test_spy_cuda(self):
        p = start_spying_on(cuda=True)
        res = []
        for i in range(0, 10):
            time.sleep(0.005)
            res.append(np.empty(i * 1000000))
        del res
        time.sleep(0.02)
        pres = p.stop()
        self.assertIsInstance(pres, dict)
        self.assertLessEqual(pres["cpu"].end, pres["cpu"].max_peak)
        self.assertLessEqual(pres["cpu"].begin, pres["cpu"].max_peak)
        self.assertIn("gpus", pres)
        self.assertLessEqual(pres["gpus"][0].end, pres["gpus"][0].max_peak)
        self.assertLessEqual(pres["gpus"][0].begin, pres["gpus"][0].max_peak)


if __name__ == "__main__":
    unittest.main(verbosity=2)
