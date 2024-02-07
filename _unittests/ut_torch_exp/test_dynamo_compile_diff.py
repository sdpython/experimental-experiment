import os
import unittest
from experimental_experiment.ext_test_case import ExtTestCase, skipif_ci_windows, ignore_warnings


class TestDynamoCompileDiff(ExtTestCase):
    @skipif_ci_windows("dynamo does not work on windows")
    @ignore_warnings((UserWarning, RuntimeWarning, DeprecationWarning))
    def test_standalone(self):
        import onnxruntime  # noqa: F401
        import logging
        import onnx
        from onnx_array_api.reference import (
            compare_onnx_execution,
            ExtendedReferenceEvaluator,
        )
        import torch
        from experimental_experiment.convert.convert_helper import (
            optimize_model_proto,
            ort_optimize,
        )
        from experimental_experiment.torch_helper.llama_helper import (
            get_llama_model,  # noqa: F401
            get_llama_attention,
            get_llama_decoder,  # noqa: F401
        )
        from experimental_experiment.torch_helper.dump_helper import (
            assert_all_close,
            dump_onnx,
            onnx_debug_backend,
            reorder_functions_in_proto,
            inputs_from_onnx_model,
            build_matching_inputs,
        )

        logging.disable(logging.ERROR)

        kwargs = dict(input_dims=[(2, 1024)] * 2)

        # if script_args.part == "attention":
        model, inputs = get_llama_attention(**kwargs)
        # model, inputs = get_llama_decoder(**kwargs)
        # model, inputs = get_llama_model(**kwargs)

        expected = model(*inputs[0])

        folder = "temp_dump_models"
        storage = {}

        optimized_mod = torch.compile(model, backend="onnxrt", fullgraph=True)
        with dump_onnx("llama_onnxrt", folder=folder, clean=True):
            expected_onnxrt = optimized_mod(*inputs[0])
        assert_all_close(expected, expected_onnxrt)

        # debugging backend
        onnx_mod = torch.compile(
            model,
            backend=lambda *args, **kwargs: onnx_debug_backend(
                *args,
                dump_prefix=os.path.join(folder, "llama_debug"),
                target_opset=18,
                storage=storage,
                **kwargs,
            ),
            fullgraph=True,
        )
        got = onnx_mod(*inputs[0])
        assert_all_close(expected, got)

        ##############################

        feeds = storage["instance"][0]["inputs"][0]

        models = os.listdir(folder)
        onnx_models = list(sorted([m for m in models if m.endswith(".onnx")]))
        assert len(onnx_models) == 2, f"unexpected value {onnx_models}"
        model_onnxrt = os.path.join(folder, onnx_models[1])
        model_debug = os.path.join(folder, onnx_models[0])
        assert inputs_from_onnx_model(model_debug)

        reorder_functions_in_proto(model_onnxrt)
        feedsrt = build_matching_inputs(model_debug, feeds, model_onnxrt)
        onnxrt = optimize_model_proto(onnx.load(model_onnxrt))
        debug = onnx.load(model_debug)

        optimized = model_onnxrt.replace(".onnx", ".opt.onnx")
        ort_optimize(onnxrt, output=optimized)
        onnxrt = onnx.load(optimized)

        optimized = model_debug.replace(".onnx", ".opt.onnx")
        ort_optimize(debug, output=optimized)
        debug = onnx.load(optimized)

        #######################

        out_onnxrt = ExtendedReferenceEvaluator(onnxrt).run(None, feedsrt)
        out_debug = ExtendedReferenceEvaluator(debug).run(None, feeds)
        assert out_onnxrt
        assert out_debug

        res1, res2, align, dc = compare_onnx_execution(
            onnxrt,
            debug,
            verbose=0,
            raise_exc=True,
            inputs=(feedsrt, feeds),
        )
        text = dc.to_str(res1, res2, align, column_size=90)
        self.assertIn("OUTPUT", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
