from typing import List
from onnx import save as onnx_save


def generate_dummy_inputs(
    model_id: str,
    num_hidden_layers: int = 2,
    n_iterations: int = -1,
    with_images: bool = True,
    device: str = "cpu",
    prefix: str = "",
    verbose: int = 0,
) -> List[str]:
    """
    Generates dummy input for a specific model.

    :param model_id: model id
    :param num_hidden_layers: the number of layers is connected to the number of caches
    :param n_iterations: number of iterations to store
    :param with_images: add images in the prompt
    :param device: device to use to run the model
    :param prefix: prefix for filename
    :param verbose: verbosity
    :return: proto
    """
    from ...mini_onnx_builder import create_onnx_model_from_input_tensors
    from ...helpers import string_type
    from .llm_dummy_inputs import create_dummy_inputs_for_phi35_vision_instruct

    written_files = []
    clean_model_id = model_id.replace("/", "_")
    prefix = f"{prefix}{clean_model_id}_{num_hidden_layers}"
    if model_id == "microsoft/Phi-3.5-vision-instruct":
        inputs = create_dummy_inputs_for_phi35_vision_instruct(
            num_hidden_layers=num_hidden_layers, with_images=with_images, device=device
        )
        if n_iterations > 0 and len(inputs) > n_iterations:
            inputs = inputs[:n_iterations]
        for i, obj in enumerate(inputs):
            filename = f"{prefix}{'.images' if with_images else ''}.iter.{i}.onnx"
            if verbose:
                print(
                    (
                        f"[generate_dummy_inputs] write {filename!r} "
                        f"with {string_type(inputs, True,True)}"
                    )
                )
            onx = create_onnx_model_from_input_tensors(obj, randomize=True)
            onnx_save(onx, filename)
            written_files.append(filename)
        return written_files

    raise NotImplementedError(
        f"Generartion of dummy inputs are not implemented for model_id={model_id}"
    )


if __name__ == "__main__":
    pass
