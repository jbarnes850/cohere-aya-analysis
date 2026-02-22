from training.build_cpt_dataset import build_cpt_dataset


def _row(lang: str, task: str, user: str, assistant: str):
    return {
        "lang": lang,
        "task": task,
        "source": "unit",
        "sample_id": f"{lang}-{task}-{user[:4]}",
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


def test_build_cpt_dataset_from_sft_rows_only():
    cfg = {
        "seed": 42,
        "data": {
            "total_texts": 20,
            "min_total_texts": 10,
            "train_ratio": 0.8,
            "min_chars": 1,
            "max_chars": 10000,
            "dedup": True,
            "quality": {"enabled": False},
            "near_dedup": {"enabled": False},
            "bucket_weights": {
                "ja_knowledge": 0.25,
                "ko_knowledge": 0.25,
                "ja_ko_instruction_mcq": 0.15,
                "en_replay": 0.25,
                "helper_transfer": 0.10,
            },
            "mcq_dev": {"enabled": False},
        },
    }

    sft_train = [
        _row("ja", "instruction", "質問A", "回答A"),
        _row("ko", "instruction", "질문B", "답변B"),
        _row("en", "instruction", "Q C", "A C"),
        _row("ja", "translation", "Translate", "訳文"),
        _row("ko", "translation", "Translate", "번역문"),
    ]
    sft_dev = [
        _row("ja", "entity", "富士通を含む文", "富士通を含む回答"),
        _row("ko", "structured", "JSON으로 답변", '{"k":"v"}'),
        _row("en", "instruction", "Q D", "A D"),
    ]

    train_rows, dev_rows, meta = build_cpt_dataset(cfg, sft_train_rows=sft_train, sft_dev_rows=sft_dev, seed=42)

    assert len(train_rows) > 0
    assert len(dev_rows) > 0
    assert meta["resolved_total_texts"] == len(train_rows) + len(dev_rows)
    assert "ja_knowledge" in meta["bucket_counts"]
    assert "ko_knowledge" in meta["bucket_counts"]


def test_build_cpt_dataset_applies_quality_and_near_dedup():
    cfg = {
        "seed": 42,
        "data": {
            "total_texts": 24,
            "min_total_texts": 12,
            "train_ratio": 0.8,
            "min_chars": 1,
            "max_chars": 10000,
            "dedup": True,
            "quality": {
                "enabled": True,
                "min_quality_score": 0.55,
                "drop_lang_mismatch": True,
                "max_repeated_line_ratio": 0.45,
                "max_repeated_char_run": 8,
                "repetition_penalty": 0.60,
                "lang_mismatch_penalty": 0.20,
            },
            "near_dedup": {
                "enabled": True,
                "hamming_threshold": 3,
                "band_bits": 16,
                "max_bucket_candidates": 32,
                "simhash_max_features": 128,
            },
            "bucket_weights": {
                "ja_knowledge": 0.25,
                "ko_knowledge": 0.25,
                "ja_ko_instruction_mcq": 0.15,
                "en_replay": 0.25,
                "helper_transfer": 0.10,
            },
            "mcq_dev": {"enabled": False},
        },
    }

    sft_train = []
    for i in range(6):
        sft_train.append(_row("ja", "instruction", f"質問{i}", f"これは日本語の高品質な回答です。詳細{i}"))
        sft_train.append(_row("ko", "instruction", f"질문{i}", f"이것은 한국어 고품질 답변입니다. 세부정보{i}"))
        sft_train.append(_row("en", "instruction", f"Question {i}", f"This is a high quality English answer with details {i}."))
    sft_train.append(_row("ja", "translation", "Translate to Korean", "한국어 번역 예시입니다."))
    sft_train.append(_row("ko", "translation", "Translate to Japanese", "日本語の翻訳例です。"))

    # Exact duplicate.
    sft_train.append(_row("ja", "instruction", "質問0", "これは日本語の高品質な回答です。詳細0"))
    # Near duplicate with tiny surface change.
    sft_train.append(_row("ja", "instruction", "質問0!", "これは日本語の高品質な回答です。詳細0"))
    # Low quality repetitive text.
    noisy_block = "\n".join(["spam spam spam"] * 30)
    sft_train.append(_row("ko", "instruction", noisy_block, noisy_block))
    # Language mismatch for JA bucket.
    sft_train.append(_row("ja", "instruction", "ミスマッチ", "This answer is in English only."))

    train_rows, dev_rows, meta = build_cpt_dataset(cfg, sft_train_rows=sft_train, sft_dev_rows=[], seed=42)
    merged = train_rows + dev_rows

    assert len(merged) >= 12
    assert all("quality_score" in row for row in merged)

    pool_quality = meta["pool_quality"]
    dropped_exact = sum(int(v.get("dropped_duplicate_exact", 0)) for v in pool_quality.values())
    dropped_near = sum(int(v.get("dropped_duplicate_near", 0)) for v in pool_quality.values())
    dropped_low_quality = sum(int(v.get("dropped_low_quality", 0)) for v in pool_quality.values())
    dropped_lang_mismatch = sum(int(v.get("dropped_lang_mismatch", 0)) for v in pool_quality.values())

    assert dropped_exact >= 1
    assert dropped_near >= 1
    assert dropped_low_quality >= 1
    assert dropped_lang_mismatch >= 1
