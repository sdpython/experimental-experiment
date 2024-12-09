"""
From `microsoft/Phi-2 <https://huggingface.co/microsoft/phi-2>`_
"""

from transformers import AutoTokenizer
from experimental_experiment.helpers import string_type
from experimental_experiment.torch_models.llm_model_helper import get_phi2

model, *_ = get_phi2(num_hidden_layers=1)
model = model.to("cpu")

# for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
model_id = "microsoft/phi-2"
print(f"-- load processor from {model_id!r}")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, torch_dtype="auto")


prompt = '''def print_prime(n):
   """
   Print all primes between 1 and n
   """
'''

inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
print(f"-- types: {string_type(inputs, with_shape=True, with_min_max=True)}")
inputs_iteration = []


def rewrite_forward(f, *args, **kwargs):
    print(f"------------- iteration {len(inputs_iteration)}")
    print(f"args: {string_type(args, with_shape=True, with_min_max=True)}")
    print(f"kwargs: {string_type(kwargs, with_shape=True, with_min_max=True)}")
    print(kwargs["input_ids"])
    inputs_iteration.append((args, kwargs))
    return f(*args, **kwargs)


print("-- intercept forward")
print(f"-- inputs type: {string_type(inputs)}")

model_forward = model.forward
model.forward = lambda f=model_forward, *args, **kwargs: rewrite_forward(f, *args, **kwargs)

outputs = model.generate(**inputs, max_length=30)

# remove input tokens
response = tokenizer.batch_decode(outputs)[0]

print("-- response", response)
print("---------------------")
print(inputs_iteration)
