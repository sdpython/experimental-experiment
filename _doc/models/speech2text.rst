=======================
Speech2Text2ForCausalLM
=======================

Dummy Example
=============

See `Speech2Text2Config
<https://huggingface.co/docs/transformers/model_doc/speech_to_text_2.Speech2Text2Config>`_.

.. code-block:: python

    import time
    import torch


    def get_speech2text2_causal_ml_not_trained_model():
        from transformers import Speech2Text2Config, Speech2Text2ForCausalLM

        config = Speech2Text2Config()
        example = torch.tensor(
            [
                [
                    0,
                    145,
                    336,
                    147,
                    147,
                    175,
                    145,
                    145,
                    3,
                    7738,
                    144,
                    336,
                    161,
                    131,
                    531,
                    160,
                    175,
                    7738,
                    114,
                    160,
                    7464,
                    2221,
                    117,
                    216,
                    160,
                    9469,
                    216,
                    9764,
                    531,
                    9570,
                    130,
                    531,
                    114,
                    160,
                    162,
                    7738,
                    114,
                    147,
                    9161,
                    114,
                    9469,
                    175,
                    9348,
                    144,
                    114,
                    336,
                    147,
                    131,
                    336,
                    147,
                    130,
                    7738,
                    114,
                    147,
                    9161,
                    166,
                    114,
                    117,
                    216,
                    147,
                    3,
                    7738,
                    175,
                    1938,
                    4626,
                    531,
                    336,
                    117,
                    336,
                    131,
                    7464,
                    336,
                    162,
                    473,
                    145,
                    145,
                    7738,
                    114,
                    160,
                    7464,
                    114,
                    7738,
                    147,
                    114,
                    131,
                    336,
                    216,
                    147,
                    114,
                    9465,
                    114,
                    7738,
                    2221,
                    312,
                    336,
                    147,
                    130,
                    1932,
                    144,
                    216,
                    175,
                    9348,
                    166,
                    336,
                    117,
                    131,
                    175,
                    9094,
                    115,
                    336,
                    160,
                    78,
                    175,
                    9469,
                    139,
                    216,
                    117,
                    131,
                    175,
                    160,
                    3,
                    7738,
                    145,
                    114,
                    147,
                    162,
                    117,
                    161,
                    114,
                    144,
                    175,
                    7738,
                    117,
                    166,
                    336,
                    145,
                    7464,
                    114,
                    9469,
                    216,
                    147,
                    7464,
                    166,
                    531,
                    161,
                    9388,
                    336,
                    9258,
                    131,
                    141,
                    7464,
                    117,
                    114,
                    166,
                    7464,
                    136,
                    114,
                    9767,
                    131,
                    141,
                    114,
                    9469,
                    166,
                    336,
                    117,
                    131,
                    175,
                    9094,
                    161,
                    114,
                    160,
                    78,
                    175,
                    9094,
                    5025,
                    175,
                    9161,
                    131,
                    1932,
                    139,
                    145,
                    114,
                    117,
                    9388,
                    141,
                    336,
                    7738,
                    131,
                    175,
                    175,
                    131,
                    9388,
                    114,
                    147,
                    9161,
                    166,
                    336,
                    117,
                    131,
                    175,
                    9094,
                    312,
                    216,
                    141,
                    9258,
                    161,
                    216,
                    145,
                    145,
                    336,
                    175,
                    9094,
                    130,
                    336,
                    293,
                    175,
                    7738,
                    141,
                    336,
                    7738,
                    117,
                    336,
                    131,
                    131,
                    175,
                    9094,
                    2221,
                    161,
                    141,
                    175,
                    175,
                    160,
                    139,
                    531,
                    9465,
                    117,
                    145,
                    114,
                    9570,
                    216,
                    9258,
                    131,
                    141,
                    7464,
                    115,
                    114,
                    161,
                    9498,
                    115,
                    175,
                    139,
                    216,
                    160,
                    7464,
                    141,
                    7464,
                    117,
                    114,
                    473,
                    7738,
                    145,
                    336,
                    78,
                    7464,
                    2221,
                    117,
                    141,
                    114,
                    166,
                    144,
                    216,
                    216,
                    175,
                    9094,
                    336,
                    9258,
                    2221,
                    131,
                    531,
                    160,
                    78,
                    336,
                    117,
                    9388,
                    115,
                    114,
                    131,
                    9388,
                    147,
                    175,
                    1938,
                    9469,
                    166,
                    114,
                ]
            ],
            dtype=torch.int64,
        )
        return (lambda: Speech2Text2ForCausalLM(config)), (example,)


    warmup = 10
    repeat = 30
    model_f, inputs = get_speech2text2_causal_ml_not_trained_model()
    model = model_f()

    # conversion to float16
    print("conversion to float16")
    model = model.to(torch.float16)
<<<<<<< HEAD
    model.eval()
=======
>>>>>>> 92a7e35948e24e271cadd9d974703feafc22352a

    # is cuda
    if torch.cuda.is_available():
        print("moves input to CUDA")
        model = model.to("cuda:1")
        inputs = tuple(i.to("cuda:1") for i in inputs)

    # warmup
<<<<<<< HEAD
    print("warmup")
=======
>>>>>>> 92a7e35948e24e271cadd9d974703feafc22352a
    for w in range(warmup):
        model(*inputs)

    # repeat
<<<<<<< HEAD
    print("repeat")
=======
>>>>>>> 92a7e35948e24e271cadd9d974703feafc22352a
    begin = time.perf_counter()
    for r in range(repeat):
        model(*inputs)
    eager = time.perf_counter() - begin
    print(f"eager: {eager}")

    # inductor
    with torch.no_grad():
        model_inductor = torch.compile(model, backend="inductor", fullgraph=True)

    # warmup inductor
<<<<<<< HEAD
    print("warmup")
=======
>>>>>>> 92a7e35948e24e271cadd9d974703feafc22352a
    for w in range(warmup):
        model_inductor(*inputs)

    # repeat
<<<<<<< HEAD
    print("repeat")
=======
>>>>>>> 92a7e35948e24e271cadd9d974703feafc22352a
    begin = time.perf_counter()
    for r in range(repeat):
        model_inductor(*inputs)
    inductor = time.perf_counter() - begin
<<<<<<< HEAD
    print(f"eager: {eager}")
    print(f"inductor: {inductor}")
    print(f"speedup: {eager / inductor}")
=======
    print(f"inductor: {inductor}")

>>>>>>> 92a7e35948e24e271cadd9d974703feafc22352a
