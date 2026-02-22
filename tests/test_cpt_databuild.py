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
