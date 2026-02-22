#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _metric(df: pd.DataFrame, metric: str, split_pattern: str) -> float:
    rows = df[(df["metric"] == metric) & (df["split"].str.contains(split_pattern, regex=True))]
    if rows.empty:
        return float("nan")
    return float(rows["value"].mean())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate post-training report markdown from metrics")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--report-path", default="docs/tiny_aya_ja_ko_frontier_report.md")
    parser.add_argument("--exec-summary-path", default="docs/tiny_aya_exec_summary.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)

    pre = pd.read_csv(run_root / "metrics" / "expanded_pre" / "metrics.csv")
    post = pd.read_csv(run_root / "metrics" / "expanded_post" / "metrics.csv")

    summary_json = run_root / "metrics" / "expanded_summary" / "decision_summary.json"
    summary = json.loads(summary_json.read_text()) if summary_json.exists() else {}

    def delta(metric: str, split: str) -> float:
        return _metric(post, metric, split) - _metric(pre, metric, split)

    ja_delta = delta("global_mmlu_accuracy", "ja")
    ko_delta = delta("global_mmlu_accuracy", "ko")
    flores_delta = delta("flores_chrfpp", ".*")

    conf_pre = _metric(pre, "language_confusion_rate", "ja|ko")
    conf_post = _metric(post, "language_confusion_rate", "ja|ko")
    conf_rel = (conf_post - conf_pre) / conf_pre if conf_pre > 0 else float("nan")

    entity_delta = delta("entity_exact_rate", "entity")
    structured_delta = delta("structured_valid_rate", "structured")

    decision = summary.get("decision", "UNKNOWN")
    frontier_gap = summary.get("frontier_gap_closure", float("nan"))

    report = f"""# Tiny Aya JA/KO Frontier Post-Training Report

## Objective
Improve Japanese and Korean quality and efficiency on Tiny Aya Global by targeting decoding-stage behavior with late-layer LoRA SFT and conditional DPO.

## Runtime
- Run root: `{run_root}`
- Decision: **{decision}**
- Frontier gap closure: **{frontier_gap:.4f}**

## Key Deltas (Expanded Eval)
- JA Global-MMLU delta: **{ja_delta:.4f}**
- KO Global-MMLU delta: **{ko_delta:.4f}**
- FLORES+ chrF++ delta: **{flores_delta:.4f}**
- Language confusion relative change (JA/KO): **{conf_rel:.4f}**
- Entity exact-rate delta: **{entity_delta:.4f}**
- Structured valid-rate delta: **{structured_delta:.4f}**

## Artifacts
- Quick pre: `{run_root / 'metrics' / 'quick_pre' / 'metrics.csv'}`
- Quick post: `{run_root / 'metrics' / 'quick_post' / 'metrics.csv'}`
- Expanded pre: `{run_root / 'metrics' / 'expanded_pre' / 'metrics.csv'}`
- Expanded post: `{run_root / 'metrics' / 'expanded_post' / 'metrics.csv'}`
- Comparator eval: `{run_root / 'metrics' / 'expanded_comparators'}`
- Summary: `{run_root / 'metrics' / 'expanded_summary' / 'decision_summary.json'}`
"""

    Path(args.report_path).write_text(report, encoding="utf-8")

    exec_summary = f"""# Tiny Aya JA/KO Frontier Executive Summary

- Decision: **{decision}**
- JA delta: **{ja_delta:.4f}**
- KO delta: **{ko_delta:.4f}**
- FLORES+ chrF++ delta: **{flores_delta:.4f}**
- Confusion relative change: **{conf_rel:.4f}**
- Frontier gap closure: **{frontier_gap:.4f}**

Recommendation: {'Proceed with rollout candidate' if decision == 'GO' else 'Run follow-up tuning iteration'}.
"""

    Path(args.exec_summary_path).write_text(exec_summary, encoding="utf-8")
    print(f"Wrote {args.report_path} and {args.exec_summary_path}")


if __name__ == "__main__":
    main()
