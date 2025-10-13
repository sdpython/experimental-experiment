import unittest
import warnings
from typing import Any
import numpy
import onnx.backend.base
import onnx.backend.test
import onnx.shape_inference
import onnx.version_converter
from onnx import ModelProto
from onnx.backend.base import Device, DeviceType
from onnx_diagnostic.helpers import string_type
from onnx_diagnostic.helpers.rt_helper import make_feeds
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.xshape.shape_builder_impl import BasicShapeBuilder
from experimental_experiment.xshape._onnx_helper import overwrite_shape_in_model_proto


class ShapeBuilderBackendRep(onnx.backend.base.BackendRep):
    def __init__(self, model: onnx.ModelProto):
        self._model = overwrite_shape_in_model_proto(model)
        self._session = ExtendedReferenceEvaluator(self._model)
        self._shape_builder = BasicShapeBuilder()

    def run(self, inputs, **kwargs):
        if isinstance(inputs, numpy.ndarray):
            inputs = [inputs]
        if isinstance(inputs, list):
            feeds = make_feeds(self._model, inputs)
        elif isinstance(inputs, dict):
            feeds = inputs
        else:
            raise TypeError(f"Unexpected input type {type(inputs)!r}.")
        outs = self._session.run(None, feeds)
        self._shape_builder.run_model(self._model)
        try:
            self._shape_builder.compare_with_true_inputs(feeds, outs, exc=True)
        except Exception as e:
            raise AssertionError(
                f"Unable to handle a model due to {str(e)}\n---\n"
                f"inputs: {string_type(feeds, with_shape=True)}\n---\n"
                f"{self._shape_builder.get_debug_msg()}\n---\n{self._model}"
            ) from e
        return outs


class ShapeBuilderBackend(onnx.backend.base.Backend):
    @classmethod
    def is_opset_supported(cls, model):  # pylint: disable=unused-argument
        return True, ""

    @classmethod
    def supports_device(cls, device: str) -> bool:
        d = Device(device)
        return d.type == DeviceType.CPU  # type: ignore[no-any-return]

    @classmethod
    def prepare(
        cls, model: onnx.ModelProto, device: str = "CPU", **kwargs: Any
    ) -> ShapeBuilderBackendRep:
        assert isinstance(model, ModelProto), f"Unexpected type {type(model)} for model."
        return ShapeBuilderBackendRep(model)

    @classmethod
    def run_model(cls, model, inputs, device=None, **kwargs):
        rep = cls.prepare(model, device, **kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return rep.run(inputs, **kwargs)


backend_test = onnx.backend.test.BackendTest(ShapeBuilderBackend())

# The following tests are too slow with the reference implementation (Conv).
backend_test.exclude(
    "(test_bvlc_alexnet|test_densenet121|test_inception_v1|test_inception_v2"
    "|test_resnet50|test_shufflenet|test_squeezenet|test_vgg19|test_zfnet512"
    "|test_bernoulli|test_gradient|test_adam_multiple|test_adagrad|test_regex_full_match"
    "|test_adam|test_if_opt|test_loop16_seq_none|test_scan_sum)"
)

# Not implemented yet.
backend_test.exclude("(affine_grid|array_feature_extractor|binarizer|label_encoder)")


# import all test cases at global scope to make them visible to python.unittest
globals().update(backend_test.test_cases)

if __name__ == "__main__":
    res = unittest.main(verbosity=2, exit=False)
    tests_run = res.result.testsRun
    errors = len(res.result.errors)
    skipped = len(res.result.skipped)
    unexpected_successes = len(res.result.unexpectedSuccesses)
    expected_failures = len(res.result.expectedFailures)
    print("---------------------------------")
    print(
        f"tests_run={tests_run} errors={errors} skipped={skipped} "
        f"unexpected_successes={unexpected_successes} "
        f"expected_failures={expected_failures}"
    )
