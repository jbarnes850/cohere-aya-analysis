from training.build_sft_dataset import (
    _build_parallel_translation_rows,
    _normalize_translation_chat_rows,
    _build_structured_rows_from_translation,
    _resolve_bucket_targets,
)


def test_resolve_bucket_targets_auto_scales_to_available_supply():
    cfg = {"data": {"target_mode": "auto_scale_to_available", "min_total_examples": 1000}}
    weights = {
        "ja_instruction": 0.35,
        "ko_instruction": 0.25,
        "translation": 0.20,
        "entity": 0.10,
        "structured": 0.10,
    }
    available = {
        "ja_instruction": 50_000,
        "ko_instruction": 2_250,
        "translation": 40_000,
        "entity": 20_000,
        "structured": 20_000,
    }

    resolved_total, targets = _resolve_bucket_targets(
        requested_total_examples=120_000,
        weights=weights,
        available_counts=available,
        cfg=cfg,
    )

    assert resolved_total <= 9_010
    assert sum(targets.values()) == resolved_total
    assert all(targets[b] <= available[b] for b in weights)


def test_build_structured_rows_from_translation_emits_required_keys():
    rows = [
        {
            "messages": [
                {"role": "user", "content": "Translate English to Korean: Hello World"},
                {"role": "assistant", "content": "안녕하세요 세계"},
            ],
            "lang": "ko",
            "task": "translation",
            "bucket": "translation",
            "source": "flores_plus",
            "translation_source_lang": "en",
            "translation_target_lang": "ko",
        }
    ]

    out = _build_structured_rows_from_translation(rows, max_rows=10, seed=42)

    assert len(out) == 1
    assert out[0]["task"] == "structured"
    assert out[0]["bucket"] == "structured"
    assert out[0]["required_keys"] == ["translation", "source_lang", "target_lang"]
    assert "\"source_lang\": \"en\"" in out[0]["messages"][-1]["content"]


def test_build_parallel_translation_rows_bidirectional():
    rows = [
        {
            "sourceString": "こんにちは",
            "targetString": "안녕하세요",
        }
    ]
    out = _build_parallel_translation_rows(
        rows=rows,
        source="tatoeba_ja_ko",
        translation_cfg={
            "source_text_field": "sourceString",
            "target_text_field": "targetString",
            "source_lang": "ja",
            "target_lang": "ko",
            "bidirectional": True,
        },
    )
    assert len(out) == 2
    pairs = {(x["translation_source_lang"], x["translation_target_lang"]) for x in out}
    assert ("ja", "ko") in pairs
    assert ("ko", "ja") in pairs


def test_normalize_translation_chat_rows_with_allowed_pairs():
    rows = [
        {
            "messages": [{"role": "user", "content": "Translate"}, {"role": "assistant", "content": "안녕하세요"}],
            "metadata": {"source_language": "en", "target_language": "ko"},
        },
        {
            "messages": [{"role": "user", "content": "Translate"}, {"role": "assistant", "content": "Bonjour"}],
            "metadata": {"source_language": "en", "target_language": "fr"},
        },
    ]
    out = _normalize_translation_chat_rows(
        rows=rows,
        source="multilingual_translation_fast",
        translation_cfg={
            "source_lang_field": "metadata.source_language",
            "target_lang_field": "metadata.target_language",
            "allowed_pairs": [["en", "ko"], ["ko", "en"]],
        },
    )
    assert len(out) == 1
    assert out[0]["translation_source_lang"] == "en"
    assert out[0]["translation_target_lang"] == "ko"
