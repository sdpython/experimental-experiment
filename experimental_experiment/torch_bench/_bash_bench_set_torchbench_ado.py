import textwrap
from ._bash_bench_set_torchbench import TorchBenchRunner


class TorchBenchAdoRunner(TorchBenchRunner):
    SUITE = "Ado"

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
    def initialize(cls):
        """Steps to run before running the benchmark."""
        expected_models = {_.strip() for _ in cls.EXPECTED_MODELS.split("\n") if _}
        cls._config = cls.load_yaml_file()
        assert "batch_size" in cls._config, f"config wrong {cls._config}"
        assert cls._config["batch_size"] is not None, f"config wrong {cls._config}"
        assert "inference" in cls._config["batch_size"], f"config wrong {cls._config}"
        for o in expected_models:
            model_name = o.strip()
            if len(model_name) < 3:
                continue
            if model_name not in cls._config["batch_size"]["inference"]:
                cls._config["batch_size"]["inference"][model_name] = 1
