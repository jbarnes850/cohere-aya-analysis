from pathlib import Path

import yaml


def test_sft_lora_layers_are_final_8():
    cfg = yaml.safe_load(Path("training/configs/tiny_aya_ja_ko_sft.yaml").read_text())
    assert cfg["lora"]["layers_to_transform"] == [28, 29, 30, 31, 32, 33, 34, 35]


def test_sft_modules_include_lm_head_save():
    cfg = yaml.safe_load(Path("training/configs/tiny_aya_ja_ko_sft.yaml").read_text())
    assert "lm_head" in cfg["lora"]["modules_to_save"]


def test_eval_configs_have_expanded_comparators():
    cfg = yaml.safe_load(Path("eval/configs/expanded_frontier.yaml").read_text())
    assert "gemma3_4b" in cfg["comparators"]
    assert "qwen3_4b" in cfg["comparators"]


def test_dpo_config_exists_and_uses_beta():
    cfg = yaml.safe_load(Path("training/configs/tiny_aya_ja_ko_dpo.yaml").read_text())
    assert cfg["dpo"]["beta"] == 0.1


def test_sft_config_has_quality_gates_and_composite_selection():
    cfg = yaml.safe_load(Path("training/configs/tiny_aya_ja_ko_sft.yaml").read_text())
    assert cfg["data"]["target_instruction_langs"] == ["ja", "ko"]
    assert cfg["data"]["source_only"] is True
    assert cfg["data"]["target_mode"] == "auto_scale_to_available"
    assert cfg["data"]["sampling_with_replacement"] is False
    assert cfg["data"]["synthetic_augmentation"]["enabled"] is False
    assert cfg["data"]["synthetic_augmentation"]["augment_translation"] is False
    assert "ko_instruction" in cfg["data"]["synthetic_augmentation"]["instruction_buckets"]
    assert cfg["datasets"]["flores_plus"]["split"] == "dev"
    assert cfg["data"]["min_total_examples"] >= 60000
    assert "llm_japanese_aya" in cfg["datasets"]
    assert "open_korean_instructions" in cfg["datasets"]
    assert "koalpaca" in cfg["datasets"]
    assert cfg["datasets"]["tatoeba_ja_ko"]["task"] == "parallel_translation"

    quality = cfg["data"]["quality"]
    assert quality["dedup"] is True
    assert quality["drop_lang_mismatch"] is True
    assert quality["max_chars_per_example"] > 0
    assert quality["min_assistant_chars"] >= 1
    assert quality["enforce_post_qc_bucket_floors"] is True
    assert quality["min_bucket_target_fraction"] >= 0.8
    assert "ko_instruction" in quality["critical_buckets"]
    assert "entity" in quality["critical_buckets"]
    assert "structured" in quality["critical_buckets"]
    assert quality["min_bucket_target_fraction_by_bucket"]["translation"] >= 0.9
    assert quality["min_bucket_target_fraction_by_bucket"]["structured"] >= 0.9

    selection = cfg["selection"]
    assert selection["enabled"] is True
    weights = selection["weights"]
    total = (
        weights["translation"]
        + weights["language_consistency"]
        + weights["entity"]
        + weights["structured"]
        + weights.get("mcq_format", 0.0)
    )
    assert abs(total - 1.0) < 1e-8

    data_weights = cfg["data"]["weights"]
    assert abs(sum(float(v) for v in data_weights.values()) - 1.0) < 1e-8


def test_expanded_global_mmlu_uses_matched_en_control():
    cfg = yaml.safe_load(Path("eval/configs/expanded_frontier.yaml").read_text())
    gm = cfg["global_mmlu"]["expanded"]
    assert gm["matched_en_control"] is True
    assert "id_fields" in gm
    assert isinstance(gm["id_fields"], list)
    assert len(gm["id_fields"]) >= 1


def test_benchmark_selection_splits_are_not_test_for_training_time():
    sft_cfg = yaml.safe_load(Path("training/configs/tiny_aya_ja_ko_sft.yaml").read_text())
    dpo_cfg = yaml.safe_load(Path("training/configs/tiny_aya_ja_ko_dpo.yaml").read_text())
    cpt_cfg = yaml.safe_load(Path("training/configs/tiny_aya_ja_ko_cpt.yaml").read_text())
    cpt_near_cfg = yaml.safe_load(Path("training/configs/tiny_aya_ja_ko_cpt_near_dedup_ablation.yaml").read_text())

    assert sft_cfg["selection"]["mcq"]["split"] == "dev"
    assert dpo_cfg["data"]["mcq_preference"]["split"] == "dev"
    assert cpt_cfg["data"]["mcq_dev"]["split"] == "dev"
    assert cpt_cfg["data"]["quality"]["enabled"] is True
    assert cpt_cfg["data"]["quality"]["min_quality_score"] >= 0.5
    assert cpt_cfg["data"]["near_dedup"]["enabled"] is False
    assert cpt_near_cfg["data"]["near_dedup"]["enabled"] is True
