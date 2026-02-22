from training.build_sft_dataset import _compute_post_qc_bucket_floor_report


def test_post_qc_bucket_floor_report_flags_failed_buckets():
    rows = [
        {"bucket": "ja_instruction"},
        {"bucket": "ja_instruction"},
        {"bucket": "ko_instruction"},
        {"bucket": "translation"},
    ]
    bucket_targets = {
        "ja_instruction": 4,
        "ko_instruction": 4,
        "translation": 2,
    }
    cfg = {
        "data": {
            "quality": {
                "enforce_post_qc_bucket_floors": True,
                "min_bucket_target_fraction": 0.5,
                "critical_buckets": ["ja_instruction", "ko_instruction", "translation"],
            }
        }
    }

    report = _compute_post_qc_bucket_floor_report(rows, bucket_targets, cfg)

    assert report["enforced"] is True
    assert report["checks"]["ja_instruction"]["status"] == "PASS"
    assert report["checks"]["translation"]["status"] == "PASS"
    assert report["checks"]["ko_instruction"]["status"] == "FAIL"
    assert "ko_instruction" in report["failed"]


def test_post_qc_bucket_floor_report_supports_per_bucket_override():
    rows = [
        {"task": "entity"},
        {"task": "entity"},
        {"task": "structured"},
    ]
    bucket_targets = {"entity": 4, "structured": 4}
    cfg = {
        "data": {
            "quality": {
                "enforce_post_qc_bucket_floors": True,
                "min_bucket_target_fraction": 0.5,
                "min_bucket_target_fraction_by_bucket": {"structured": 0.25},
                "critical_buckets": ["entity", "structured"],
            }
        }
    }

    report = _compute_post_qc_bucket_floor_report(rows, bucket_targets, cfg)

    assert report["checks"]["entity"]["status"] == "PASS"
    assert report["checks"]["structured"]["status"] == "PASS"
    assert report["failed"] == {}
