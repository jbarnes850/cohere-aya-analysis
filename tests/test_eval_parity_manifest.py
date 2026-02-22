from eval.run_eval_suite import _build_eval_parity_manifest


class _DummyTokenizer:
    chat_template = "<chat>"
    pad_token = "<pad>"
    pad_token_id = 0
    eos_token = "</s>"
    eos_token_id = 2


def test_eval_parity_manifest_includes_required_fields():
    cfg = {
        "global_mmlu": {
            "quick": {"n_shot": 5, "max_new_tokens": 8, "temperature": 0.0, "top_p": 1.0},
        },
        "flores": {
            "quick": {"max_new_tokens": 256, "temperature": 0.0, "top_p": 1.0},
        },
        "aya_eval": {
            "quick": {"max_new_tokens": 256, "temperature": 0.0, "top_p": 1.0},
        },
        "custom": {
            "quick": {"entity_max_new_tokens": 96, "structured_max_new_tokens": 128, "temperature": 0.0, "top_p": 1.0},
        },
    }

    manifest = _build_eval_parity_manifest(
        cfg=cfg,
        mode="quick",
        tokenizer=_DummyTokenizer(),
        attn_implementation="flash_attention_2",
    )

    assert manifest["mode"] == "quick"
    assert manifest["sampling"]["global_mmlu"]["n_shot"] == 5
    assert manifest["sampling"]["global_mmlu"]["generation_mode"] == "greedy"
    assert manifest["sampling"]["flores"]["max_new_tokens"] == 256
    assert manifest["special_tokens"]["chat_template_present"] is True
    assert manifest["model_runtime"]["attn_implementation_arg"] == "flash_attention_2"
