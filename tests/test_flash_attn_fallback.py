from unittest.mock import patch

import training.train_dpo as dpo
import training.train_sft as sft


class _DummyModel:
    def __init__(self) -> None:
        self.gradient_checkpointing_enabled = False

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing_enabled = True


def _sft_cfg() -> dict:
    return {
        "model": {
            "base_model": "models/tiny-aya-global",
            "attn_implementation": "flash_attention_2",
        },
        "training": {"bf16": True, "gradient_checkpointing": True},
        "fallback": {"qlora": {"bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": True}},
    }


def _dpo_cfg() -> dict:
    return {
        "model": {
            "base_model": "models/tiny-aya-global",
            "attn_implementation": "flash_attention_2",
        },
        "training": {"bf16": True},
        "lora": {
            "r": 64,
            "lora_alpha": 128,
            "lora_dropout": 0.05,
            "bias": "none",
            "target_modules": ["q_proj"],
            "layers_to_transform": [1],
            "modules_to_save": ["lm_head"],
        },
    }


def test_sft_loader_uses_sdpa_when_flash_attn_missing():
    cfg = _sft_cfg()
    with (
        patch("training.train_sft.importlib.util.find_spec", return_value=None),
        patch("training.train_sft.AutoModelForCausalLM.from_pretrained", return_value=_DummyModel()) as mock_load,
    ):
        model = sft._load_model(cfg)
    assert mock_load.call_args.kwargs["attn_implementation"] == "sdpa"
    assert model.gradient_checkpointing_enabled is True


def test_sft_loader_retries_with_sdpa_on_importerror():
    cfg = _sft_cfg()
    calls = []

    def _side_effect(*args, **kwargs):
        calls.append(kwargs["attn_implementation"])
        if len(calls) == 1:
            raise ImportError("flash-attn not available")
        return _DummyModel()

    with (
        patch("training.train_sft.importlib.util.find_spec", return_value=object()),
        patch("training.train_sft.AutoModelForCausalLM.from_pretrained", side_effect=_side_effect),
    ):
        sft._load_model(cfg)
    assert calls == ["flash_attention_2", "sdpa"]


def test_dpo_loader_uses_sdpa_when_flash_attn_missing():
    cfg = _dpo_cfg()
    with (
        patch("training.train_dpo.importlib.util.find_spec", return_value=None),
        patch("training.train_dpo.AutoModelForCausalLM.from_pretrained", return_value=_DummyModel()) as mock_load,
        patch("training.train_dpo.get_peft_model", side_effect=lambda model, _cfg: model),
    ):
        dpo._load_model(cfg, sft_adapter_dir=None)
    assert mock_load.call_args.kwargs["attn_implementation"] == "sdpa"


def test_dpo_loader_retries_with_sdpa_on_importerror():
    cfg = _dpo_cfg()
    calls = []

    def _side_effect(*args, **kwargs):
        calls.append(kwargs["attn_implementation"])
        if len(calls) == 1:
            raise ImportError("flash-attn not available")
        return _DummyModel()

    with (
        patch("training.train_dpo.importlib.util.find_spec", return_value=object()),
        patch("training.train_dpo.AutoModelForCausalLM.from_pretrained", side_effect=_side_effect),
        patch("training.train_dpo.get_peft_model", side_effect=lambda model, _cfg: model),
    ):
        dpo._load_model(cfg, sft_adapter_dir=None)
    assert calls == ["flash_attention_2", "sdpa"]
