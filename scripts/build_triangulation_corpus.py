#!/usr/bin/env python3
"""Build the E1A triangulation corpus from FLORES+ devtest.

The corpus contains 112 prompts: 4 cross-language pairs × 14 source_row_ids
× 2 sides per pair (Reading 1, bidirectional). Schema matches the v2 enterprise
packet's translation_calibration rows, with two extra fields (pair_label, side)
to identify the bidirectional factorial structure.

Pairs (X-Y), source row indices 0-13 in FLORES+ devtest:

  Pair    Side A (X-source -> Y-target)      Side B (Y-source -> X-target)
  JA-EN   jpn_Jpan -> eng_Latn               eng_Latn -> jpn_Jpan
  KO-EN   kor_Hang -> eng_Latn               eng_Latn -> kor_Hang
  AR-EN   arb_Arab -> eng_Latn               eng_Latn -> arb_Arab
  JA-Zh   jpn_Jpan -> cmn_Hans               cmn_Hans -> jpn_Jpan

(FLORES+ uses cmn_Hans for Simplified Chinese; the v2 packet's report references
 zho_Hans interchangeably. Both denote Mandarin in Simplified Han script.)

Prompt template matches v2 packet exactly:
  "Translate the following text into <Target>. Return only the translation.\n\n
   Text: <source>\n\nTranslation:"

Output: data/triangulation_corpus_v1/rows.jsonl (112 rows).

Usage:
  python3 scripts/build_triangulation_corpus.py [--output PATH] [--n-rows N]
  python3 scripts/build_triangulation_corpus.py --inspect [N]   # dump first N pairs

The HF dataset 'openlanguagedata/flores_plus' is loaded with each language as a
separate config; row N in the devtest split for any language is parallel by
construction (FLORES+ is fully parallel).

This script is CPU-only and idempotent. Re-running overwrites the output file.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


LOGGER = logging.getLogger("build_triangulation_corpus")

DEFAULT_OUTPUT = Path("data/triangulation_corpus_v1/rows.jsonl")
DEFAULT_N_ROWS = 14

# FLORES+ language codes (BCP-47 plus script tag).
LANG_CODE: Dict[str, str] = {
    "JA": "jpn_Jpan",
    "KO": "kor_Hang",
    "EN": "eng_Latn",
    "AR": "arb_Arab",
    "Zh": "cmn_Hans",
}

# Display name used in the prompt instruction (matches v2 packet conventions).
LANG_NAME: Dict[str, str] = {
    "JA": "Japanese",
    "KO": "Korean",
    "EN": "English",
    "AR": "Arabic",
    "Zh": "Chinese",
}

# Pairs to build, ordered (pair_label, X, Y).
PAIRS: List[tuple] = [
    ("JA-EN", "JA", "EN"),
    ("KO-EN", "KO", "EN"),
    ("AR-EN", "AR", "EN"),
    ("JA-Zh", "JA", "Zh"),
]

PROMPT_TEMPLATE = (
    "Translate the following text into {target}. Return only the translation."
    "\n\nText: {source}\n\nTranslation:"
)


@dataclass(frozen=True)
class CorpusRow:
    packet_row_id: str
    pair_label: str
    side: str  # "A" or "B"
    source_row_id: str
    source_lang: str  # short label, e.g., "JA"
    target_lang: str  # short label, e.g., "EN"
    source_text: str
    prompt: str
    language: str  # target language code, matches v2 packet 'language' field

    def to_jsonl(self) -> str:
        return json.dumps(
            {
                "packet_id": "triangulation_corpus_v1",
                "packet_row_id": self.packet_row_id,
                "pair_label": self.pair_label,
                "side": self.side,
                "source_row_id": self.source_row_id,
                "source_lang_code": LANG_CODE[self.source_lang],
                "target_lang_code": LANG_CODE[self.target_lang],
                "source_lang": self.source_lang,
                "target_lang": self.target_lang,
                "source_text": self.source_text,
                "prompt": self.prompt,
                "language": LANG_CODE[self.target_lang],
                "config": LANG_CODE[self.target_lang],
                "eval_only": True,
            },
            ensure_ascii=False,
        )


def load_flores_devtest(lang_code: str, n_rows: int) -> List[str]:
    """Load the first n_rows source-text strings from FLORES+ devtest for lang_code.

    Tries datasets.load_dataset; falls back to a clearer error if the dataset
    can't be reached. Returns a list of raw text strings (no instruction wrapping).
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "The `datasets` package is required. Install with: uv pip install datasets"
        ) from e

    LOGGER.info("Loading FLORES+ devtest for %s ...", lang_code)
    ds = load_dataset(
        "openlanguagedata/flores_plus",
        lang_code,
        split="devtest",
    )
    if len(ds) < n_rows:
        raise SystemExit(
            f"FLORES+ devtest for {lang_code} has only {len(ds)} rows, "
            f"need at least {n_rows}."
        )
    texts = [ds[i]["text"] for i in range(n_rows)]
    LOGGER.info("  loaded %d rows for %s", len(texts), lang_code)
    return texts


def build_corpus(n_rows: int) -> List[CorpusRow]:
    """Build the bidirectional corpus rows for all four pairs."""
    # Determine the unique set of source languages we need.
    needed: set = set()
    for _, x, y in PAIRS:
        needed.add(x)
        needed.add(y)

    # Load FLORES+ devtest for each needed language.
    texts: Dict[str, List[str]] = {}
    for short in sorted(needed):
        code = LANG_CODE[short]
        texts[short] = load_flores_devtest(code, n_rows)

    # Build prompts.
    rows: List[CorpusRow] = []
    for pair_label, x, y in PAIRS:
        for i in range(n_rows):
            # Side A: X-source -> Y-target (e.g., JA->EN: source=jpn, target=Translate into English)
            side_a_prompt = PROMPT_TEMPLATE.format(
                target=LANG_NAME[y], source=texts[x][i]
            )
            rows.append(
                CorpusRow(
                    packet_row_id=f"{pair_label.lower()}-A-flores-{i}",
                    pair_label=pair_label,
                    side="A",
                    source_row_id=str(i),
                    source_lang=x,
                    target_lang=y,
                    source_text=texts[x][i],
                    prompt=side_a_prompt,
                    language=LANG_CODE[y],
                )
            )
            # Side B: Y-source -> X-target (the reverse direction)
            side_b_prompt = PROMPT_TEMPLATE.format(
                target=LANG_NAME[x], source=texts[y][i]
            )
            rows.append(
                CorpusRow(
                    packet_row_id=f"{pair_label.lower()}-B-flores-{i}",
                    pair_label=pair_label,
                    side="B",
                    source_row_id=str(i),
                    source_lang=y,
                    target_lang=x,
                    source_text=texts[y][i],
                    prompt=side_b_prompt,
                    language=LANG_CODE[x],
                )
            )

    return rows


def write_jsonl(rows: List[CorpusRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(r.to_jsonl())
            f.write("\n")
    LOGGER.info("Wrote %d rows to %s", len(rows), output_path)


def inspect(rows: List[CorpusRow], n_show: int = 4) -> None:
    """Print first n_show rows for human verification."""
    print(f"\n=== Corpus inspection (first {n_show} pairs) ===")
    pairs_seen: set = set()
    shown = 0
    for r in rows:
        key = (r.pair_label, r.source_row_id)
        if key in pairs_seen:
            continue
        pairs_seen.add(key)
        # find both sides
        sides = [
            row for row in rows if row.pair_label == r.pair_label
            and row.source_row_id == r.source_row_id
        ]
        if len(sides) != 2:
            continue
        side_a = next(s for s in sides if s.side == "A")
        side_b = next(s for s in sides if s.side == "B")
        print(f"\n--- {r.pair_label} | source_row_id={r.source_row_id} ---")
        print(f"  Side A ({side_a.source_lang}->{side_a.target_lang})")
        print(f"    source ({len(side_a.source_text)} chars): {side_a.source_text[:120]!r}")
        print(f"    prompt last 80 chars: {side_a.prompt[-80:]!r}")
        print(f"  Side B ({side_b.source_lang}->{side_b.target_lang})")
        print(f"    source ({len(side_b.source_text)} chars): {side_b.source_text[:120]!r}")
        print(f"    prompt last 80 chars: {side_b.prompt[-80:]!r}")
        shown += 1
        if shown >= n_show:
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--n-rows", type=int, default=DEFAULT_N_ROWS,
        help=f"Number of FLORES devtest rows to use per pair (default: {DEFAULT_N_ROWS})",
    )
    parser.add_argument(
        "--inspect", action="store_true",
        help="Print first 4 pairs for verification (does not write file unless combined with --write)",
    )
    parser.add_argument(
        "--write", action="store_true",
        help="When used with --inspect, also write the file",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args()

    rows = build_corpus(n_rows=args.n_rows)
    LOGGER.info(
        "Built %d corpus rows: %d pairs × %d source rows × 2 sides",
        len(rows), len(PAIRS), args.n_rows,
    )

    if args.inspect:
        inspect(rows)
        if args.write:
            write_jsonl(rows, args.output)
        return 0

    write_jsonl(rows, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
