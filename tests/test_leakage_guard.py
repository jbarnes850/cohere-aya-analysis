from training.build_sft_dataset import _validate_no_flores_split_overlap as _validate_builder_overlap
from training.run_frontier_pipeline import _validate_no_flores_split_overlap


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
