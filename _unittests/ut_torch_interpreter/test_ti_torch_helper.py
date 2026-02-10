import unittest
import numpy as np
import torch
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.torch_interpreter._torch_helper import _tune_thresholds_histc


class TestTorchInterpreterTorchHelper(ExtTestCase):
    def test__tune_thresholds_histc(self):
        dtype, ibins, fmin, fmax = np.float16, 20, -5, 5

        # thresholds
        delta = (float(fmax) - float(fmin)) / float(ibins)
        ctype = np.float16
        delta = np.array(delta, dtype=ctype)
        min = np.array(fmin, dtype=ctype)
        max = np.array(fmax, dtype=ctype)
        bins = int(ibins)
        thresholds = np.zeros((bins + 1,), dtype=dtype)
        halfway = bins + 1 - (bins + 1) // 2
        for i in range(halfway):
            thresholds[i] = min + delta * i
        for i in range(halfway, bins + 1):
            thresholds[i] = max - delta * (bins - i)
        thresholds[-1] = np.nextafter(
            thresholds[-1], np.array(np.inf, dtype=np.float16), dtype=np.float16
        )
        thresholds = torch.from_numpy(thresholds)

        # tuning
        thresholds = _tune_thresholds_histc(thresholds, bins=ibins, fmin=fmin, fmax=fmax)

        # inputs
        keep = []
        hh = np.array(0.999, dtype=np.float16)
        for _ in range(20):
            keep.append(float(hh))
            hh = np.nextafter(hh, np.array(np.inf, dtype=np.float16), dtype=np.float16)

        inputs = [
            (torch.tensor(keep[::-1], dtype=torch.float16).reshape((-1, 10)),),
            (torch.tensor(keep, dtype=torch.float16).reshape((-1, 10)),),
            ((torch.tensor(keep[::-1], dtype=torch.float16) + 0.5).reshape((-1, 10)),),
            ((torch.tensor(keep, dtype=torch.float16) + 0.5).reshape((-1, 10)),),
            ((torch.tensor(keep[::-1], dtype=torch.float16) - 0.5).reshape((-1, 10)),),
            ((torch.tensor(keep, dtype=torch.float16) - 0.5).reshape((-1, 10)),),
            ((torch.tensor(keep[::-1], dtype=torch.float16) - 1).reshape((-1, 10)),),
            ((torch.tensor(keep, dtype=torch.float16) - 1).reshape((-1, 10)),),
        ]

        # checks
        for xs in inputs:
            x = xs[0]
            expected = torch.histc(x, bins=ibins, min=fmin, max=fmax)

            value = thresholds.unsqueeze(1) < x.reshape((-1,)).unsqueeze(0)
            value = value.sum(dim=1).squeeze()
            res = value[:-1] - value[1:]
            res = res.to(torch.float16)
            if torch.abs(expected - res).max() > 0:
                raise AssertionError(f"ERROR\n{expected=}\n-----{res=}\n{x=}\n{thresholds=}")
            self.assertEqualArray(expected, res)


if __name__ == "__main__":
    unittest.main(verbosity=2)
