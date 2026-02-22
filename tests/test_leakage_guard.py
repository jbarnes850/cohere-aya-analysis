from training.build_sft_dataset import _validate_no_flores_split_overlap as _validate_builder_overlap
from training.run_frontier_pipeline import _validate_no_flores_split_overlap, _validate_training_split_safety
from src.frontier_utils import assert_no_benchmark_test_split


def test_flores_split_overlap_raises():
    sft_cfg = {
        "datasets": {
            "flores_plus": {
                "path": "openlanguagedata/flores_plus",
                "split": "devtest",
            }
        }
    }
    quick_cfg = {
        "flores": {
            "quick": {"dataset_id": "openlanguagedata/flores_plus", "split": "devtest"},
            "expanded": {"dataset_id": "openlanguagedata/flores_plus", "split": "devtest"},
        }
    }
    expanded_cfg = quick_cfg

    try:
        _validate_no_flores_split_overlap(sft_cfg, quick_cfg, expanded_cfg)
        assert False, "expected RuntimeError for overlap"
    except RuntimeError as exc:
        assert "FLORES split leakage detected" in str(exc)

    try:
        _validate_builder_overlap(sft_cfg, quick_cfg, expanded_cfg)
        assert False, "expected RuntimeError for overlap"
    except RuntimeError as exc:
        assert "FLORES split leakage detected" in str(exc)


def test_flores_split_disjoint_passes():
    sft_cfg = {
        "datasets": {
            "flores_plus": {
                "path": "openlanguagedata/flores_plus",
                "split": "dev",
            }
        }
    }
    quick_cfg = {
        "flores": {
            "quick": {"dataset_id": "openlanguagedata/flores_plus", "split": "devtest"},
            "expanded": {"dataset_id": "openlanguagedata/flores_plus", "split": "devtest"},
        }
    }
    expanded_cfg = quick_cfg
    _validate_no_flores_split_overlap(sft_cfg, quick_cfg, expanded_cfg)


def test_benchmark_test_split_guard_rejects_heldout():
    try:
        assert_no_benchmark_test_split(
            dataset_id="CohereLabs/Global-MMLU-Lite",
            split="test",
            purpose="unit_test",
        )
        assert False, "expected ValueError for test split"
    except ValueError as exc:
        assert "Benchmark leakage guard triggered" in str(exc)


def test_benchmark_test_split_guard_allows_dev():
    assert_no_benchmark_test_split(
        dataset_id="CohereLabs/Global-MMLU-Lite",
        split="dev",
        purpose="unit_test",
    )


def test_training_split_safety_fails_on_test_split():
    sft_cfg = {
        "selection": {
            "mcq": {
                "enabled": True,
                "dataset_id": "CohereLabs/Global-MMLU-Lite",
                "split": "test",
            }
        }
    }
    pref_cfg = {
        "data": {
            "mcq_preference": {
                "enabled": True,
                "dataset_id": "CohereLabs/Global-MMLU-Lite",
                "split": "dev",
            }
        }
    }
    cpt_cfg = {
        "data": {
            "mcq_dev": {
                "enabled": True,
                "dataset_id": "CohereLabs/Global-MMLU-Lite",
                "split": "dev",
            }
        }
    }
    try:
        _validate_training_split_safety(sft_cfg, pref_cfg, cpt_cfg)
        assert False, "expected ValueError for unsafe SFT MCQ split"
    except ValueError as exc:
        assert "Benchmark leakage guard triggered" in str(exc)


def test_training_split_safety_passes_on_dev_splits():
    sft_cfg = {
        "selection": {
            "mcq": {
                "enabled": True,
                "dataset_id": "CohereLabs/Global-MMLU-Lite",
                "split": "dev",
            }
        }
    }
    pref_cfg = {
        "data": {
            "mcq_preference": {
                "enabled": True,
                "dataset_id": "CohereLabs/Global-MMLU-Lite",
                "split": "dev",
            }
        }
    }
    cpt_cfg = {
        "data": {
            "mcq_dev": {
                "enabled": True,
                "dataset_id": "CohereLabs/Global-MMLU-Lite",
                "split": "dev",
            }
        }
    }
    _validate_training_split_safety(sft_cfg, pref_cfg, cpt_cfg)
