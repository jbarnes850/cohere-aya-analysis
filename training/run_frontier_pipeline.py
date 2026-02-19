#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.frontier_utils import ensure_dir, load_yaml, now_utc_iso, timestamp_run_id


def _run(cmd: List[str]) -> None:
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _minutes_remaining(start_time: float, core_budget_minutes: float) -> float:
    elapsed = (time.time() - start_time) / 60.0
    return core_budget_minutes - elapsed


def _collect_flores_eval_splits(eval_cfg: Dict[str, Any]) -> set[str]:
    flores_cfg = eval_cfg.get("flores", {})
    splits: set[str] = set()
    for mode in ["quick", "expanded"]:
        mode_cfg = flores_cfg.get(mode, {})
        dataset_id = str(mode_cfg.get("dataset_id", "")).strip().lower()
        split = str(mode_cfg.get("split", "")).strip()
        if dataset_id.endswith("flores_plus") and split:
            splits.add(split)
    return splits


def _validate_no_flores_split_overlap(
    sft_cfg: Dict[str, Any],
    quick_eval_cfg: Dict[str, Any],
    expanded_eval_cfg: Dict[str, Any],
) -> None:
    train_splits: set[str] = set()
    for ds_cfg in sft_cfg.get("datasets", {}).values():
        if not isinstance(ds_cfg, dict):
            continue
        dataset_path = str(ds_cfg.get("path", "")).strip().lower()
        split = str(ds_cfg.get("split", "")).strip()
        if dataset_path.endswith("flores_plus") and split:
            train_splits.add(split)

    if not train_splits:
        return

    eval_splits = _collect_flores_eval_splits(quick_eval_cfg) | _collect_flores_eval_splits(expanded_eval_cfg)
    overlap = sorted(train_splits & eval_splits)
    if overlap:
        raise RuntimeError(
            "FLORES split leakage detected between training and evaluation. "
            f"overlapping_splits={overlap}. "
            "Use a disjoint training split (for example train on dev and evaluate on devtest)."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute Tiny Aya JA/KO frontier training+eval plan")
    parser.add_argument("--sft-config", default="training/configs/tiny_aya_ja_ko_sft.yaml")
    parser.add_argument("--pref-config", default="training/configs/tiny_aya_ja_ko_dpo.yaml")
    parser.add_argument("--quick-eval-config", default="eval/configs/quick_8h.yaml")
    parser.add_argument("--expanded-eval-config", default="eval/configs/expanded_frontier.yaml")
    parser.add_argument("--output-root", default="outputs/posttrain")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--core-budget-hours", type=float, default=8.0)
    parser.add_argument("--skip-expanded", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()
    python_exec = sys.executable
    sft_cfg = load_yaml(args.sft_config)
    quick_eval_cfg = load_yaml(args.quick_eval_config)
    expanded_eval_cfg = load_yaml(args.expanded_eval_config)
    _validate_no_flores_split_overlap(sft_cfg, quick_eval_cfg, expanded_eval_cfg)
    base_model_id = str(sft_cfg["model"]["base_model"])

    run_id = args.run_id or timestamp_run_id("tiny_aya_ja_ko_frontier")
    run_root = ensure_dir(Path(args.output_root) / run_id)
    metrics_root = ensure_dir(run_root / "metrics")
    artifacts_root = ensure_dir(run_root / "artifacts")

    state: Dict[str, Any] = {
        "run_id": run_id,
        "started_at_utc": now_utc_iso(),
    }

    # Phase A: build SFT dataset
    _run(
        [
            python_exec,
            "-m",
            "training.build_sft_dataset",
            "--config",
            args.sft_config,
            "--output-dir",
            str(Path(args.output_root)),
            "--run-id",
            run_id,
        ]
    )
    data_dir = run_root / "artifacts" / "data"

    # Phase B: baseline quick eval
    _run(
        [
            python_exec,
            "-m",
            "eval.run_eval_suite",
            "--config",
            args.quick_eval_config,
            "--mode",
            "quick",
            "--model-id",
            base_model_id,
            "--model-label",
            "tiny_aya_base",
            "--output-dir",
            str(metrics_root / "quick_pre"),
        ]
    )

    # Phase C: SFT
    _run(
        [
            python_exec,
            "-m",
            "training.train_sft",
            "--config",
            args.sft_config,
            "--data-dir",
            str(data_dir),
            "--run-id",
            run_id,
            "--output-root",
            str(Path(args.output_root)),
        ]
    )
    sft_adapter = run_root / "artifacts" / "sft" / "adapter"

    # Phase D: DPO conditional
    final_adapter = sft_adapter
    remaining = _minutes_remaining(start, args.core_budget_hours * 60)
    if remaining >= 90:
        _run(
            [
                python_exec,
                "-m",
                "training.build_pref_dataset",
                "--config",
                args.pref_config,
                "--sft-train-jsonl",
                str(data_dir / "train.jsonl"),
                "--sft-dev-jsonl",
                str(data_dir / "dev.jsonl"),
                "--output-dir",
                str(Path(args.output_root)),
                "--run-id",
                run_id,
            ]
        )

        pref_dir = run_root / "artifacts" / "pref_data"
        _run(
            [
                python_exec,
                "-m",
                "training.train_dpo",
                "--config",
                args.pref_config,
                "--pref-data-dir",
                str(pref_dir),
                "--sft-adapter-dir",
                str(sft_adapter),
                "--remaining-minutes",
                str(remaining),
                "--run-id",
                run_id,
                "--output-root",
                str(Path(args.output_root)),
            ]
        )
        pref_adapter = run_root / "artifacts" / "dpo" / "adapter"
        if pref_adapter.exists():
            final_adapter = pref_adapter

    # Phase E: post quick eval
    _run(
        [
            python_exec,
            "-m",
            "eval.run_eval_suite",
            "--config",
            args.quick_eval_config,
            "--mode",
            "quick",
            "--model-id",
            base_model_id,
            "--model-label",
            "tiny_aya_post",
            "--adapter-dir",
            str(final_adapter),
            "--output-dir",
            str(metrics_root / "quick_post"),
        ]
    )

    # Expanded eval + comparators (can exceed 8h)
    if not args.skip_expanded:
        _run(
            [
                python_exec,
                "-m",
                "eval.run_eval_suite",
                "--config",
                args.expanded_eval_config,
                "--mode",
                "expanded",
                "--model-id",
                base_model_id,
                "--model-label",
                "tiny_aya_base",
                "--output-dir",
                str(metrics_root / "expanded_pre"),
            ]
        )

        _run(
            [
                python_exec,
                "-m",
                "eval.run_eval_suite",
                "--config",
                args.expanded_eval_config,
                "--mode",
                "expanded",
                "--model-id",
                base_model_id,
                "--model-label",
                "tiny_aya_post",
                "--adapter-dir",
                str(final_adapter),
                "--output-dir",
                str(metrics_root / "expanded_post"),
            ]
        )

        _run(
            [
                python_exec,
                "-m",
                "eval.run_comparators",
                "--config",
                args.expanded_eval_config,
                "--mode",
                "expanded",
                "--output-root",
                str(metrics_root / "expanded_comparators"),
            ]
        )

        _run(
            [
                python_exec,
                "-m",
                "eval.compare_pre_post",
                "--pre-metrics",
                str(metrics_root / "expanded_pre" / "metrics.csv"),
                "--post-metrics",
                str(metrics_root / "expanded_post" / "metrics.csv"),
                "--comparators-root",
                str(metrics_root / "expanded_comparators"),
                "--output-dir",
                str(metrics_root / "expanded_summary"),
            ]
        )

    state["completed_at_utc"] = now_utc_iso()
    state["base_model_id"] = base_model_id
    state["preference_stage"] = "dpo"
    state["final_adapter"] = str(final_adapter)
    state["core_minutes_elapsed"] = round((time.time() - start) / 60.0, 2)

    with open(artifacts_root / "pipeline_state.json", "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
