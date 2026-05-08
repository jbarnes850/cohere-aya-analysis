#!/usr/bin/env python3
"""Build the E2 SAE activation training corpus.

Sources (in priority order):
  1. v2 enterprise packet (100 rows) — includes 36 Marco-MIF, 28 FLORES,
     36 single-language enterprise. Most directly aligned with the route
     window we want SAE to factor.
  2. FLORES+ devtest (1012 parallel sentences in each of 5 languages =
     5060 sentences) — clean parallel multilingual baseline.
  3. FLORES+ dev (997 parallel sentences × 5 = 4985 sentences) — additional
     parallel multilingual data, optional.

The corpus is text-only. Activation extraction (GPU step) wraps each text
through the model's tokenizer and forwards.

Estimated activation count (per model × layer):
  - With v2 + FLORES devtest only (~5160 prompts × ~50 tokens avg) ≈ 250K activations.
  - With v2 + FLORES devtest + FLORES dev (~10K prompts × ~50 tokens) ≈ 500K activations.
  - 500K activations is exploratory-scale (production SAEs use 1B+).

Schema (output JSONL):
  {
    "source": "v2-packet" | "flores-devtest" | "flores-dev",
    "row_id": str,                    # source row identifier
    "language": str,                  # language code (BCP-47-ish)
    "text": str,                      # text to forward through model
    "is_enterprise_prompt": bool,     # True if from v2 packet (with task framing)
    "task_family": Optional[str],     # ja-flores / ko-mif-json_format / ...
  }

Output: data/sae_activation_corpus_v1/rows.jsonl.

Usage:
  uv run python scripts/build_sae_activation_corpus.py [--include-flores-dev]
  uv run python scripts/build_sae_activation_corpus.py --inspect [N]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List


LOGGER = logging.getLogger("build_sae_activation_corpus")

DEFAULT_OUTPUT = Path("data/sae_activation_corpus_v1/rows.jsonl")

# Same five languages as E1A.
FLORES_LANGS = ["jpn_Jpan", "kor_Hang", "eng_Latn", "arb_Arab", "cmn_Hans"]


def load_v2_packet_rows(packet_path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in packet_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        # Each v2 packet row is an enterprise prompt.
        # Extract the task family from packet_row_id.
        pid = r["packet_row_id"]
        if pid.startswith("ja-flores-"):
            family = "ja-flores"
        elif pid.startswith("ko-flores-"):
            family = "ko-flores"
        elif pid.startswith("ja-mif-"):
            family = "ja-mif"
        elif pid.startswith("ko-mif-"):
            family = "ko-mif"
        elif pid.startswith("ja-datapilot-"):
            family = "ja-datapilot"
        elif pid.startswith("ko-law-"):
            family = "ko-law"
        elif pid.startswith("ko-legal-qa-"):
            family = "ko-legal-qa"
        else:
            family = "unknown"
        rows.append({
            "source": "v2-packet",
            "row_id": pid,
            "language": r.get("language", "unknown"),
            "text": r["prompt"],
            "is_enterprise_prompt": True,
            "task_family": family,
        })
    return rows


def load_flores_split(split: str) -> List[Dict[str, Any]]:
    """Load FLORES+ `split` for all 5 languages and return flat row list."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "datasets is required. Install with: uv pip install datasets"
        ) from e
    rows = []
    for lang in FLORES_LANGS:
        LOGGER.info("Loading FLORES+ %s for %s ...", split, lang)
        ds = load_dataset("openlanguagedata/flores_plus", lang, split=split)
        for i in range(len(ds)):
            rows.append({
                "source": f"flores-{split}",
                "row_id": f"{lang}-{split}-{i}",
                "language": lang,
                "text": ds[i]["text"],
                "is_enterprise_prompt": False,
                "task_family": None,
            })
        LOGGER.info("  loaded %d rows for %s", len(ds), lang)
    return rows


def write_jsonl(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")
    LOGGER.info("Wrote %d rows to %s", len(rows), output_path)


def inspect(rows: List[Dict[str, Any]], n_show: int = 8) -> None:
    print("\n=== SAE activation corpus inspection ===")
    print(f"Total rows: {len(rows)}")
    by_source: Dict[str, int] = {}
    by_lang: Dict[str, int] = {}
    by_family: Dict[str, int] = {}
    total_chars = 0
    for r in rows:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
        by_lang[r["language"]] = by_lang.get(r["language"], 0) + 1
        if r.get("task_family"):
            by_family[r["task_family"]] = by_family.get(r["task_family"], 0) + 1
        total_chars += len(r["text"])
    print(f"Avg text length (chars): {total_chars / max(len(rows), 1):.0f}")
    print("\nBy source:")
    for k, v in sorted(by_source.items()):
        print(f"  {k}: {v}")
    print("\nBy language (top 10):")
    for k, v in sorted(by_lang.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {k}: {v}")
    if by_family:
        print("\nBy enterprise task family:")
        for k, v in sorted(by_family.items()):
            print(f"  {k}: {v}")
    print(f"\nFirst {n_show} samples:")
    for i, r in enumerate(rows[:n_show]):
        text_preview = r["text"][:100].replace("\n", " ")
        print(
            f"  [{i}] source={r['source']:>15} lang={r['language']:>9} "
            f"family={str(r.get('task_family','-')):>15} "
            f"text({len(r['text'])} chars): {text_preview!r}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--packet", type=Path, required=True,
                        help="JSONL path; row schema documented in README")
    parser.add_argument(
        "--include-flores-dev", action="store_true",
        help="Include FLORES+ dev split (additional 997 × 5 sentences) on top of devtest",
    )
    parser.add_argument(
        "--inspect", action="store_true",
        help="Print summary and first 8 rows; do not write file unless --write",
    )
    parser.add_argument(
        "--write", action="store_true",
        help="With --inspect, also write the file",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args()

    LOGGER.info("Loading v2 enterprise packet from %s", args.packet)
    rows = load_v2_packet_rows(args.packet)
    LOGGER.info("Loaded %d v2 enterprise packet rows", len(rows))

    LOGGER.info("Loading FLORES+ devtest (5 languages × 1012)")
    rows += load_flores_split("devtest")

    if args.include_flores_dev:
        LOGGER.info("Loading FLORES+ dev (5 languages × 997)")
        rows += load_flores_split("dev")

    LOGGER.info("Total corpus rows: %d", len(rows))

    if args.inspect:
        inspect(rows)
        if args.write:
            write_jsonl(rows, args.output)
        return 0

    write_jsonl(rows, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
