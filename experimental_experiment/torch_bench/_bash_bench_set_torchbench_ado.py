import textwrap
from ._bash_bench_set_torchbench import TorchBenchRunner


class TorchBenchAdoRunner(TorchBenchRunner):

    EXPECTED_MODELS = textwrap.dedent(
        """
        codellama
        hf_distil_whisper
        hf_mixtral
        hf_Yi
        llama_v2_7b_16h
        moondream
        llava
        mistral_7b_instruct
        orca_2
        phi_1_5
        phi_2
        stable_diffusion_text_encoder
        stable_diffusion_unet
        stable_diffusion_xl
        """
    )

    @classmethod
    def initialize(container):
        """
        Steps to run before running the benchmark.
        """
        expected_models = set(
            _.strip() for _ in container.EXPECTED_MODELS.split("\n") if _
        )
        container._config = container.load_yaml_file()
        assert "batch_size" in container._config, f"config wrong {container._config}"
        assert (
            container._config["batch_size"] is not None
        ), f"config wrong {container._config}"
        assert (
            "inference" in container._config["batch_size"]
        ), f"config wrong {container._config}"
        for o in expected_models:
            model_name = o.strip()
            if len(model_name) < 3:
                continue
            if model_name not in container._config["batch_size"]["inference"]:
                container._config["batch_size"]["inference"][model_name] = 1
