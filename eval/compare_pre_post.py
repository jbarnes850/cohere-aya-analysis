#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.frontier_utils import ensure_dir, now_utc_iso


HIGHER_IS_BETTER = {
    "global_mmlu_accuracy": True,
    "flores_chrfpp": True,
    "language_confusion_rate": False,
    "entity_exact_rate": True,
    "structured_valid_rate": True,
}


def _load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["metric", "split", "model", "value"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {path}")
    return df


def _collect_comparator_metrics(comparators_root: Path) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for metrics_file in comparators_root.glob("*/metrics.csv"):
        rows.append(_load_metrics(metrics_file))
    if not rows:
        raise ValueError(f"No comparator metrics found under {comparators_root}")
    return pd.concat(rows, ignore_index=True)


def _best_comparator(df: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    for (metric, split), group in df.groupby(["metric", "split"]):
        hib = HIGHER_IS_BETTER.get(metric, True)
        idx = group["value"].idxmax() if hib else group["value"].idxmin()
        row = group.loc[idx].copy()
        row["best_comparator_value"] = row["value"]
        out_rows.append(row[["metric", "split", "model", "best_comparator_value"]])
    return pd.DataFrame(out_rows)


def _gap_closure(metric: str, baseline: float, post: float, best: float) -> float:
    hib = HIGHER_IS_BETTER.get(metric, True)
    if hib:
        baseline_gap = best - baseline
        new_gap = best - post
    else:
        baseline_gap = baseline - best
        new_gap = post - best

    if baseline_gap <= 0:
        return np.nan
    return (baseline_gap - new_gap) / baseline_gap


def _aggregate(df: pd.DataFrame, metric: str, split_filter: str) -> float:
    if not split_filter:
        split_mask = pd.Series([True] * len(df), index=df.index)
    else:
        split_mask = df["split"].str.contains(split_filter, regex=False, na=False)
    m = df[(df["metric"] == metric) & split_mask]
    if m.empty:
        return np.nan
    return float(m["value"].mean())


def _compute_go_no_go(merged: pd.DataFrame) -> Dict[str, Dict[str, float | str]]:
    def val(metric: str, split: str, col: str, use_regex: bool = False) -> float:
        if not split:
            split_mask = pd.Series([True] * len(merged), index=merged.index)
        else:
            split_mask = merged["split"].str.contains(split, regex=use_regex, na=False)
        sub = merged[(merged["metric"] == metric) & split_mask]
        if sub.empty:
            return float("nan")
        return float(sub[col].mean())

    checks: Dict[str, Dict[str, float | str]] = {}

    ja_mmlu_delta = val("global_mmlu_accuracy", "ja", "delta")
    ko_mmlu_delta = val("global_mmlu_accuracy", "ko", "delta")
    flores_delta = val("flores_chrfpp", "", "delta")
    confusion_baseline = val("language_confusion_rate", "ja|ko", "baseline_value", use_regex=True)
    confusion_post = val("language_confusion_rate", "ja|ko", "post_value", use_regex=True)
    confusion_rel = ((confusion_post - confusion_baseline) / confusion_baseline) if confusion_baseline > 0 else float("nan")
    entity_delta = val("entity_exact_rate", "entity", "delta")
    structured_delta = val("structured_valid_rate", "structured", "delta")
    en_regression = val("global_mmlu_accuracy", "en", "delta")

    checks["ja_global_mmlu"] = {
        "value": ja_mmlu_delta,
        "target": 0.015,
        "status": "PASS" if ja_mmlu_delta >= 0.015 else "FAIL",
    }
    checks["ko_global_mmlu"] = {
        "value": ko_mmlu_delta,
        "target": 0.015,
        "status": "PASS" if ko_mmlu_delta >= 0.015 else "FAIL",
    }
    checks["flores_chrfpp"] = {
        "value": flores_delta,
        "target": 2.0,
        "status": "PASS" if flores_delta >= 2.0 else "FAIL",
    }
    checks["confusion_relative"] = {
        "value": confusion_rel,
        "target": -0.25,
        "status": "PASS" if confusion_rel <= -0.25 else "FAIL",
    }
    checks["entity_delta"] = {
        "value": entity_delta,
        "target": 0.10,
        "status": "PASS" if entity_delta >= 0.10 else "FAIL",
    }
    checks["structured_delta"] = {
        "value": structured_delta,
        "target": 0.08,
        "status": "PASS" if structured_delta >= 0.08 else "FAIL",
    }
    checks["en_guardrail"] = {
        "value": en_regression,
        "target": -0.005,
        "status": "PASS" if en_regression >= -0.005 else "FAIL",
    }

    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare pre/post metrics and compute frontier gap closure")
    parser.add_argument("--pre-metrics", required=True)
    parser.add_argument("--post-metrics", required=True)
    parser.add_argument("--comparators-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pre_df = _load_metrics(Path(args.pre_metrics))
    post_df = _load_metrics(Path(args.post_metrics))
    cmp_df = _collect_comparator_metrics(Path(args.comparators_root))

    baseline = pre_df[["metric", "split", "value"]].rename(columns={"value": "baseline_value"})
    post = post_df[["metric", "split", "value"]].rename(columns={"value": "post_value"})
    best = _best_comparator(cmp_df)

    merged = baseline.merge(post, on=["metric", "split"], how="inner")
    merged = merged.merge(best[["metric", "split", "best_comparator_value"]], on=["metric", "split"], how="left")
    merged["delta"] = merged["post_value"] - merged["baseline_value"]
    merged["gap_closure"] = merged.apply(
        lambda r: _gap_closure(r["metric"], r["baseline_value"], r["post_value"], r["best_comparator_value"])
        if pd.notna(r["best_comparator_value"])
        else np.nan,
        axis=1,
    )

    # Combined JA/KO frontier scorecard
    focus = merged[
        merged["metric"].isin(["global_mmlu_accuracy", "flores_chrfpp", "language_confusion_rate", "entity_exact_rate", "structured_valid_rate"])
    ]
    focus_jako = focus[
        focus["split"].str.contains("ja", case=False) | focus["split"].str.contains("ko", case=False) | focus["split"].str.contains("entity") | focus["split"].str.contains("structured")
    ]
    if focus_jako.empty:
        frontier_gap_closure = 0.0
    else:
        non_nan_gap = focus_jako["gap_closure"].dropna()
        frontier_gap_closure = float(non_nan_gap.mean()) if not non_nan_gap.empty else 0.0

    checks = _compute_go_no_go(merged)
    checks["frontier_gap_closure"] = {
        "value": frontier_gap_closure,
        "target": 0.50,
        "status": "PASS" if frontier_gap_closure >= 0.50 else "FAIL",
    }

    n_fail = sum(1 for c in checks.values() if c["status"] == "FAIL")
    core_keys = [
        "ja_global_mmlu",
        "ko_global_mmlu",
        "flores_chrfpp",
        "confusion_relative",
        "entity_delta",
        "structured_delta",
        "en_guardrail",
    ]
    core_pass = all(checks[k]["status"] == "PASS" for k in core_keys)
    frontier_value = float(checks["frontier_gap_closure"]["value"])

    if core_pass and frontier_value >= 0.50:
        decision = "GO"
    elif core_pass and frontier_value >= 0.30:
        decision = "CONDITIONAL_GO"
    else:
        decision = "NO_GO"

    out_dir = ensure_dir(args.output_dir)
    merged.to_csv(out_dir / "pre_post_comparison.csv", index=False)

    checks_df = pd.DataFrame(
        [{"check": k, **v} for k, v in checks.items()]
    )
    checks_df.to_csv(out_dir / "go_no_go_checks.csv", index=False)

    report = {
        "completed_at_utc": now_utc_iso(),
        "decision": decision,
        "failed_checks": int(n_fail),
        "frontier_gap_closure": frontier_gap_closure,
        "checks": checks,
    }
    with open(out_dir / "decision_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_lines = [
        "# Frontier Comparison Summary",
        "",
        f"- Decision: **{decision}**",
        f"- Frontier gap closure: **{frontier_gap_closure:.3f}**",
        f"- Failed checks: **{n_fail}**",
        "",
        "## Checks",
        "",
        "| Check | Value | Target | Status |",
        "|---|---:|---:|---|",
    ]
    for check, payload in checks.items():
        v = payload["value"]
        t = payload["target"]
        md_lines.append(f"| {check} | {v:.4f} | {t:.4f} | {payload['status']} |")

    (out_dir / "decision_summary.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
