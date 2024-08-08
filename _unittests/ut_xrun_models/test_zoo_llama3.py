import sys
import unittest
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    requires_torch,
    requires_zoo,
)


class TestZooLlama3(ExtTestCase):

    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @ignore_warnings(DeprecationWarning)
    @requires_torch("2.3")
    @requires_zoo()  # ZOO=1 python _unittests/ut_xrun_models/test_zoo_llama3.py
    def test_llama3_export(self):

        import os
        import time
        import onnxruntime
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from experimental_experiment.torch_interpreter import to_onnx

        model_id = "meta-llama/Meta-Llama-3-8B"

        with torch.no_grad():
            model = AutoModelForCausalLM.from_pretrained(model_id).eval()
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            base_prompt = "Is the conversion to onnx going to work?"
            base_inputs = tokenizer(base_prompt, return_tensors="pt")  # .to("cpu")
            input_ids = base_inputs.input_ids
            expected = model(input_ids)

            print(f"output type: {type(expected)}")
            print(f"logits: {expected.logits.shape}, {expected.logits.dtype}")

            print(
                "start conversion... with input_ids", input_ids.dtype, input_ids.shape
            )
            begin = time.perf_counter()
            large_onx = to_onnx(
                model,
                (input_ids,),
                input_names=["x"],
                verbose=1,
                large_model=True,
                # dynamic_shapes fails with transformers==4.37.2
                # TypeError: scaled_dot_product_attention():
                # argument 'is_causal' must be bool, not SymBool
                # dynamic_shapes={"x": {1: torch.export.Dim("length", min=2)}},
            )
            duration = time.perf_counter() - begin
            print(f"conversion done in {duration}s")

        folder = "test_zoo_export_llama3"
        if not os.path.exists(folder):
            os.mkdir(folder)
        else:
            for _ in os.listdir(folder):
                os.remove(os.path.join(folder, _))

        print(f"start saving in {folder!r}")
        begin = time.perf_counter()
        filename = os.path.join(folder, "llama3.onnx")
        large_onx.save(filename)
        duration = time.perf_counter() - begin
        print(f"saving done in {duration}s with {len(os.listdir(folder))} files")

        print(f"loading model {filename!r} with onnxruntime.")
        begin = time.perf_counter()
        sess = onnxruntime.InferenceSession(
            filename, providers=["CPUExecutionProvider"]
        )
        print(f"done in {time.perf_counter() - begin}s")

        print("running the first iteration")
        begin = time.perf_counter()
        name = large_onx.model_proto.graph.input[0].name
        np_input = input_ids.detach().cpu().numpy()
        got = sess.run(None, {name: np_input})
        print(f"done in {time.perf_counter() - begin}s")
        self.assertEqualArray(expected.logits, got[0], atol=1e-4)

        N = 5
        print(f"running {N} iterations with torch")
        begin = time.perf_counter()
        for i in range(N):
            model(input_ids)
        d = time.perf_counter() - begin
        print(f"done in {d}s for torch")

        print(f"running {N} iterations with onnxruntime")
        begin = time.perf_counter()
        for i in range(N):
            sess.run(None, {name: np_input})
        d = time.perf_counter() - begin
        print(f"done in {d}s for onnxruntime")


if __name__ == "__main__":
    unittest.main(verbosity=2)
