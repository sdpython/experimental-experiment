"""
From `microsoft/Phi-3.5-vision-instruct
<https://huggingface.co/microsoft/Phi-3.5-vision-instruct>`_
"""

from PIL import Image
import requests
from transformers import AutoModelForCausalLM, AutoProcessor
from experimental_experiment.helpers import string_type

model_id = "microsoft/Phi-3.5-vision-instruct"

print(f"-- load model from {model_id!r}")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    trust_remote_code=True,
    torch_dtype="auto",
    # _attn_implementation='flash_attention_2'
    # flash_attn only works for GPU >= H100
    _attn_implementation="eager",
)

# for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
print(f"-- load processor from {model_id!r}")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, num_crops=4)

images = []
placeholder = ""

# Note: if OOM, you might consider reduce number of frames in this example.
for i in range(1, 3):
    url = f"https://image.slidesharecdn.com/azureintroduction-191206101932/75/Introduction-to-Microsoft-Azure-Cloud-{i}-2048.jpg"
    print(f"-- download image from {url!r}")
    img = Image.open(requests.get(url, stream=True).raw)
    print(f"   size: {img.size}")
    images.append(img)
    placeholder += f"<|image_{i}|>\n"

messages = [
    {"role": "user", "content": placeholder + "Summarize the deck of slides."},
]

prompt = processor.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
print("------------------- prompt")
print(prompt)
print("------------------- end prompt")

print("-- create inputs")
inputs = processor(prompt, images, return_tensors="pt").to("cuda:0")
print(f"-- types: {string_type(inputs, with_shape=True, with_min_max=True)}")
print(f"-- image_sizes {inputs.image_sizes}")
# BatchFeature(data=dict(input_ids:T7s1x777[-1:32010],
#                        attention_mask:T7s1x777[1:1],
#                        pixel_values:T1s1x5x3x336x336[-2.063474178314209:2.305359125137329],
#                        image_sizes:T7s1x2[672:672]))

generation_args = {
    "max_new_tokens": 30,
    "temperature": 0.0,
    "do_sample": False,
}


def rewrite_forward(f, *args, **kwargs):
    print("forward input:")
    print(f"args: {string_type(args, with_shape=True, with_min_max=True)}")
    print(f"kwargs: {string_type(kwargs, with_shape=True, with_min_max=True)}")
    print(kwargs["input_ids"])
    return f(*args, **kwargs)


print("-- intercept forward")
model_forward = model.forward
model.forward = lambda f=model_forward, *args, **kwargs: rewrite_forward(f, *args, **kwargs)

generate_ids = model.generate(
    **inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args
)

# remove input tokens
generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
response = processor.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

print("-- response")
# args: ()
# kwargs: dict(input_ids:T7s1x777[-1:32010],
#              position_ids:T7s1x777[0:776],
#              past_key_values:DynamicCache,
#                   #32[  T16s1x32x875x96[-8.375:6.5625], T16s1x32x875x96[-8.9375:8.875] ]
#              use_cache:int, True
#              attention_mask:T7s1x777, (boolean)
#              pixel_values:T1s1x5x3x336x336,
#              image_sizes:T7s1X2,
#              return_dict:int)
# kwargs: dict(input_ids:T7s1x777[-1:32010],position_ids:T7s1x777[0:776],
# past_key_values:DynamicCache(key_cache=[], DynamicCache(value_cache=[]),use_cache:int[True],
# attention_mask:T7s1x777[1:1],
# pixel_values:T1s1x5x3x336x336[-2.063474178314209:2.305359125137329],
# image_sizes:T7s1x2[672:672],return_dict:int[True])
print(response)
