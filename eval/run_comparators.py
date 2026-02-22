#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.frontier_utils import ensure_dir, load_yaml, now_utc_iso

from eval.run_eval_suite import run_eval


def _try_model_ids(candidates: List[str], *args, **kwargs):
    last_exc = None
    for model_id in candidates:
        try:
            result = run_eval(*args, model_id=model_id, **kwargs)
            return model_id, result
        except Exception as exc:
            print(f"WARN: comparator {model_id} failed: {exc}")
            last_exc = exc
    raise RuntimeError(f"All comparator candidates failed: {candidates}") from last_exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run expanded comparator inference")
    parser.add_argument("--config", default="eval/configs/expanded_frontier.yaml")
    parser.add_argument("--mode", choices=["quick", "expanded"], default="expanded")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    comparators: Dict[str, List[str]] = cfg["comparators"]
    out_root = ensure_dir(Path(args.output_root))

    results = {
        "started_at_utc": now_utc_iso(),
        "mode": args.mode,
        "runs": [],
    }

    for label, candidate_ids in comparators.items():
        model_id, summary = _try_model_ids(
            candidate_ids,
            config_path=args.config,
            mode=args.mode,
            model_label=label,
            adapter_dir=None,
            output_dir=str(out_root / label),
            attn_implementation=args.attn_implementation,
        )
        results["runs"].append({"label": label, "resolved_model_id": model_id, "summary": summary})

    results["completed_at_utc"] = now_utc_iso()
    with open(out_root / "comparators_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
