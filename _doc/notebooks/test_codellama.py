import os
import pprint
import torch

# see https://huggingface.co/docs/transformers/en/model_doc/code_llama
if os.path.exists("CodeLlama-7b-model"):
    print("load the model")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained("./CodeLlama-7b-tokenizer")
    model = AutoModelForCausalLM.from_pretrained("./CodeLlama-7b-model")
else:
    print("retrieve the model")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
    tokenizer.save_pretrained('CodeLlama-7b-tokenizer')
    model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-hf")
    model.save_pretrained('CodeLlama-7b-model')

print("done")

# move the model to cuda
model = model.to("cuda")


PROMPT = '''def remove_non_ascii(s: str) -> str:

    """ <FILL_ME>

    return result

'''
with torch.no_grad():
    input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
    input_ids = input_ids.to("cuda")
    print("run the model")
    generated_ids = model.generate(input_ids, max_new_tokens=128).to("cuda")
    print("interpret the answer")
    filling = tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens = True)[0]
    print("---")
    print(PROMPT.replace("<FILL_ME>", filling))
    print("done")

import time

times = []

N = 10
if True:
    with torch.no_grad():
        # warmup
        print("warmup")
        for _ in range(3):
            model(input_ids, use_cache=False)

        # repeat
        print("repeat")
        begin = time.perf_counter()
        for _ in range(N):
            model(input_ids, use_cache=False)
        d = (time.perf_counter() - begin) / N
        times.append(("eager", d))
        print("avg time eager", d)


# export to onnx
from experimental_experiment.torch_interpreter import to_onnx
from experimental_experiment.xbuilder import OptimizationOptions
from experimental_experiment.torch_dynamo import onnx_custom_backend

if not os.path.exists("codellama.onnx"):
    with torch.no_grad():
        onx = to_onnx(model, (input_ids,), optimize=True, large_model=True, options=OptimizationOptions(patterns="default"))
        onx.save("codellama.onnx", all_tensors_to_one_file=True)


for optim in ["default+onnxruntime"]:#, "default+onnxruntime", "default"]:
    with torch.no_grad():
        options = OptimizationOptions(
            constant_folding=True,
            patterns=optim,
            verbose=0,
            processor="CUDA",
        )

        custom_custom_backend = lambda *args, **kwargs: onnx_custom_backend(  # noqa: E731
            *args,
            target_opset=18,
            verbose=0,
            options=options,
            optimize=True,
            dump_prefix=f"dump_onx_code_llama_{optim.replace('+', '_')}",
            **kwargs,
        )    

        compiled_model = torch.compile(model, backend=custom_custom_backend, fullgraph=True, dynamic=False)

        # warmup
        print("warmup compiled model")
        for _ in range(3):
            compiled_model(input_ids, use_cache=False)

        # repeat
        print("repeat compiled_model")
        begin = time.perf_counter()
        for _ in range(N):
            compiled_model(input_ids, use_cache=False)
        d = (time.perf_counter() - begin) / N
        times.append((optim, d))
        print(f"avg time custom backend with optimization={optim!r}", d)

print("-----------------")
pprint.pprint(times)
