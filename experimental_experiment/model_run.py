import time
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from onnx import ModelProto, TensorProto


def create_tensor(shape: Tuple[int, ...], dtype: int, batch_size: int = 1) -> np.ndarray:
    """
    Creates a random tensor.

    :param shape: shape
    :param dtype: onnx type
    :param batch_size: batch_size
    :return: numpy array
    """
    assert (
        not isinstance(shape, int) or shape[0] == batch_size
    ), f"Conflict between batch_size={batch_size} and shape={shape}"
    if not isinstance(shape[0], int):
        shape = (batch_size, *shape[1:])
    t = np.random.random(shape)
    if dtype in (TensorProto.FLOAT, "tensor(float)"):
        return t.astype(np.float32)
    if dtype == (TensorProto.FLOAT16, "tensor(float16)"):
        return t.astype(np.float16)
    if dtype in (TensorProto.DOUBLE, "tensor(double)"):
        return t.astype(np.float64)

    raise AssertionError(f"The function is not implemented yet for dtype={dtype!r}")


def create_feeds(
    sess: "onnxruntime.InferenceSession", batch_size: int = 1  # noqa: F821
) -> Dict[str, Any]:
    """
    Creates random feeds for a model.

    :param sess: onnxruntime session
    :param batch_size: batch_size
    :return: feeds
    """
    feeds = {}
    for inp in sess.get_inputs():
        feeds[inp.name] = create_tensor(inp.shape, inp.type, batch_size=batch_size)
    return feeds


def model_run(
    model: Union[str, ModelProto],
    repeat: int = 10,
    warmup: int = 5,
    batch_size: int = 1,
    processor: str = "CPU",
    verbose: int = 0,
    validate: Optional[Union[str, ModelProto]] = None,
) -> Dict[str, Any]:
    """
    Loads a model with onnxruntime and measures the inference time.

    :param model: model to run
    :param warmup: number of iterations to run before measuring
    :param repeat: number of iterations to run to measure
    :param batch_size: batch size of the inputs
    :param processor: processor to run
    :param verbose: verbosity
    :return: metrics
    """
    if processor == "CPU":
        providers = ["CPUExecutionProvider"]
    elif processor == "CUDA":
        providers = ["CUDAExecutionProvider"]
    elif processor == "CUDA,CPU":
        providers = ["CPUExecutionProvider", "CUDAExecutionProvider"]
    else:
        raise AssertionError(f"Unexpected value {processor!r} for processor.")

    assert (
        processor == "CPU"
    ), f"This function does not implement anything yet for CUDA, processor={processor!r}"

    if verbose:
        smodel = model if isinstance(model, str) else str(type(model))
        print(f"[model_run] loads {smodel!r}")
        print(f"[model_run] providers={providers}")

    stats = {
        "providers": providers,
        "model": model,
        "repeat": repeat,
        "warmup": warmup,
        "batch_size": batch_size,
    }

    begin = time.perf_counter()
    from onnxruntime import InferenceSession

    stats["time_import_ort"] = time.perf_counter() - begin

    if verbose:
        print(f"[model_run] import ort {stats['time_import_ort']!r}")

    feeds = None

    if validate is not None and validate != "":
        from onnx_diagnostic.helpers import max_diff

        if verbose:
            smodel = validate if isinstance(validate, str) else str(type(validate))
            print(f"[model_run] validate with {smodel!r}")

        begin = time.perf_counter()
        val = InferenceSession(
            validate.SerializeToString() if isinstance(validate, ModelProto) else validate,
            providers=providers,
        )
        stats["time_ort_load_validation"] = time.perf_counter() - begin
        if feeds is None:
            begin = time.perf_counter()
            feeds = create_feeds(val, batch_size=batch_size)
            stats["time_ort_create_feeds"] = time.perf_counter() - begin

        if verbose:
            print("[model_run] compute expected output")
        begin = time.perf_counter()
        expected = val.run(None, feeds)
        stats["time_latency_validation_one"] = time.perf_counter() - begin

    begin = time.perf_counter()
    model_bytes = model.SerializeToString() if isinstance(model, ModelProto) else model
    stats["time_model_bytes"] = time.perf_counter() - begin

    if verbose:
        print(f"[model_run] time model bytes {stats['time_model_bytes']!r}")

    begin = time.perf_counter()
    sess = InferenceSession(model_bytes, providers=providers)
    stats["time_ort_load"] = time.perf_counter() - begin

    if verbose:
        print(f"[model_run] time ort load {stats['time_ort_load']!r}")
        print("[model_run] create inputs")

    if feeds is None:
        begin = time.perf_counter()
        feeds = create_feeds(sess, batch_size=batch_size)
        stats["time_ort_create_feeds"] = time.perf_counter() - begin

    for k, v in feeds.items():
        stats[f"onnx_input_{k}_shape"] = "x".join(map(str, v.shape))
        stats[f"onnx_input_{k}_dtype"] = str(v.dtype).split(".")[-1]

    if verbose:
        print(f"[model_run] create {len(feeds)} inputs")
        for k, v in feeds.items():
            print(f"[model_run] input {k!r}: {v.dtype} - {v.shape}")
        print(f"[model_run] warmup {warmup} times")

    begin = time.perf_counter()
    for _ in range(warmup):
        last = sess.run(None, feeds)
    stats["time_warmup_total"] = time.perf_counter() - begin
    stats["time_warmup"] = stats["time_warmup_total"] / float(warmup)

    if validate is not None and validate != "":
        if verbose:
            print("[model_run] compare outputs")
        disc = max_diff(expected, last)
        for k, v in disc.items():
            stats[f"discrepancies_{k}"] = v
            if verbose:
                print(f"[model_run] discrepancies {k}: {v}")

    if verbose:
        print(f"[model_run] warmup took {stats['time_warmup_total']}")
        print(f"[model_run] repeat {repeat} times")

    times = []
    for _ in range(repeat):
        begin = time.perf_counter()
        sess.run(None, feeds)
        times.append(time.perf_counter() - begin)
    np_times = np.array(times)
    stats["time_latency_total"] = np_times.sum()
    stats["time_latency_t_std"] = np_times.std()
    stats["time_latency_t_min"] = np_times.min()
    stats["time_latency_t_max"] = np_times.max()
    stats["time_latency_t_med"] = np.median(np_times)
    # measure the correlation with the time
    stats["time_latency_t_corrt"] = np.corrcoef(np_times, list(range(len(times))))[0, 1]
    # measures the correlation with the previous value
    stats["time_latency_t_corrp"] = np.corrcoef(np_times[1:], np_times[:-1])[0, 1]
    stats["time_latency"] = stats["time_latency_total"] / len(times)
    stats["time_latency_t_qu"] = "/".join(map(str, np.quantile(np_times, np.arange(11) / 10.0)))

    if verbose:
        print(f"[model_run] inference took {stats['time_latency_total']}")
        print("[model_run] done")

    return stats
