import unittest
from experimental_experiment.ext_test_case import ExtTestCase, hide_stdout
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.torch_interpreter import to_onnx
from experimental_experiment.torch_interpreter.investigate_helper import run_aligned


class TestInvestigateHelper(ExtTestCase):

    @hide_stdout()
    def test_ep_onnx_sync(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                ry = x.abs()
                rz = ry.exp()
                rw = rz + 1
                ru = rw.log() + rw
                return ru

        x = torch.randn((5, 4))
        Model()(x)
        ep = torch.export.export(Model(), (x,), dynamic_shapes=({0: torch.export.Dim("batch")},))
        onx = to_onnx(ep)
        results = list(
            run_aligned(
                ep,
                onx,
                (x,),
                check_conversion_cls=dict(cls=ExtendedReferenceEvaluator, atol=1e-5, rtol=1e-5),
                verbose=1,
            ),
        )
        self.assertEqual(len(results), 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
