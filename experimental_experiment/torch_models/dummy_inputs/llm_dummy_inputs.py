def get_dummy_inputs_for_phi_3_5_vision_instruct(
    num_hidden_layers: int = 2, device: str = "cpu", verbose: int = 0
):
    """
    Generates dummy inputs for an untrained model using the same
    structure as Phi 3.5 instruct vision.
    """

    from PIL import Image
    import requests
    from transformers import AutoProcessor
    from ...helpers import string_type
    from ...torch_models.llm_model_helper import get_phi_3_5_vision_instruct

    model, _ = get_phi_3_5_vision_instruct(num_hidden_layers=num_hidden_layers)
    model = model.to(device)

    model_id = "microsoft/Phi-3.5-vision-instruct"
    if verbose:
        print(
            f"[get_dummy_inputs_for_phi_3_5_vision_instruct] "
            f"get processor for model {model_id!r}"
        )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, num_crops=4)

    images = []
    placeholder = ""

    for i in range(1, 3):
        url = f"https://image.slidesharecdn.com/azureintroduction-191206101932/75/Introduction-to-Microsoft-Azure-Cloud-{i}-2048.jpg"
        if verbose:
            print(
                f"[get_dummy_inputs_for_phi_3_5_vision_instruct] download image from {url!r}"
            )
        img = Image.open(requests.get(url, stream=True).raw)
        print(f"[get_dummy_inputs_for_phi_3_5_vision_instruct] image size {img.size}")
        images.append(img)
        placeholder += f"<|image_{i}|>\n"

    messages = [
        {"role": "user", "content": placeholder + "Summarize the deck of slides."},
    ]

    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    if verbose:
        print("[get_dummy_inputs_for_phi_3_5_vision_instruct] create inputs")
    inputs = processor(prompt, images, return_tensors="pt").to(device)
    if verbose:
        print(
            f"[get_dummy_inputs_for_phi_3_5_vision_instruct] types: "
            f"{string_type(inputs, with_shape=True, with_min_max=True)}"
        )
        print(
            f"[get_dummy_inputs_for_phi_3_5_vision_instruct] image_sizes {inputs.image_sizes}"
        )

    generation_args = {"max_new_tokens": 30, "temperature": 0.0, "do_sample": False}

    inputs_iteration = []

    def rewrite_forward(f, *args, **kwargs):
        inputs_iteration.append((args, kwargs))
        return f(*args, **kwargs)

    model_forward = model.forward
    model.forward = lambda f=model_forward, *args, **kwargs: rewrite_forward(
        f, *args, **kwargs
    )

    generate_ids = model.generate(
        **inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args
    )
    assert generate_ids is not None

    # generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
    # response = processor.batch_decode(
    #     generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    # )[0]

    return inputs_iteration
