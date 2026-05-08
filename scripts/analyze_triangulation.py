#!/usr/bin/env python3
"""Experiment E1: three-axis triangulation of the layer-28-34 commit window.

Pre-registration: docs/experiment_e1_e2_prereg.md.

Sub-experiments (all use last-token residual at every layer index 0..36):

  E1A — Cross-language translation triangulation (NEW corpus):
    4 pairs (JA-EN, KO-EN, AR-EN, JA-Zh), 14 FLORES devtest rows, bidirectional
    Side A / Side B. Within-model factorial: B = cos(A[i], B[i]),
    D = cos(A[i], B[(i+7)%14]); analogous C and E. Tests cross-language
    generalization of the cosine-geometry route window.

  E1B — Cross-task instruction-following triangulation within JA-KO (v2 packet):
    17 paired Marco-MIF JA-KO source_row_ids. Two analysis lenses: primary
    (8 content-parallel) and secondary (all 17). Tests cross-task generalization
    within JA-KO.

  E1C — Cross-checkpoint single-language enterprise geometry (v2 packet):
    36 unpaired enterprise rows (12 ja-datapilot, 12 ko-law, 12 ko-legal-qa).
    Per-row cross-checkpoint cosine cos(h_base[i, L], h_post[i, L]) with
    post in {Global, Water}. Tests whether post-training reorganization
    happens at layers 28-34 for single-language enterprise tasks.

Layer convention (pinned across the codebase):
  output_hidden_states=True returns a tuple of length n_layers + 1 = 37 for
  Tiny Aya. Index 0 is post-embedding; indices 1..36 are post-block residual
  streams (layer L for L>=1 is the residual after transformer block L-1).

Anchor token:
  All E1A and E1B prompts end with "Translation:" or another ":" anchor;
  E1C prompts end with whatever the v2 packet provides. The cosine is
  always computed at the last token, so anchor identity is asserted in
  --render-only mode for E1A/E1B and reported (not asserted) for E1C.

Modes:
  --render-only            CPU; tokenizer-only sanity for E1A and E1B prompts.
  --mde-sanity             CPU; synthetic random hidden states; bootstrap CI
                            half-widths at N in {8, 14, 17}.
  default (--extract)      GPU; forward passes for ONE model, save hidden
                            states to <output>/<slug>/e1{a,b,c}_hidden_states.npy
                            plus metadata CSVs and within-model cosine CSVs.
  --analyze                CPU; read hidden states from all 3 models,
                            compute cross-model summaries (E1C) and aggregate
                            E1A/E1B verdicts.

Outputs (per model):
  <output>/<run_tag>/experiment_e1/<slug>/
    e1a_hidden_states.npy        shape (112, 37, 2048), float32
    e1a_metadata.csv             prompt_idx -> (pair_label, side, source_row_id, ...)
    e1a_per_layer_cosine.csv     long-format B/C/D/E per pair per layer
    e1a_delta_meaning_summary.csv
    e1b_hidden_states.npy        shape (34, 37, 2048), float32
    e1b_metadata.csv
    e1b_per_layer_cosine.csv
    e1b_delta_meaning_summary.csv
    e1c_hidden_states.npy        shape (36, 37, 2048), float32
    e1c_metadata.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Make `src.*` importable when invoked as `python scripts/run_experiment_e1_triangulation.py`
# from the workspace root. `python script.py` only adds the script's own directory to
# sys.path; we need the workspace root (parent of scripts/) so that `from src.logit_lens
# import ...` resolves on both local and GPU machines.
_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))


LOGGER = logging.getLogger("experiment_e1_triangulation")

DEFAULT_TRIANG_CORPUS = Path("data/triangulation_corpus_v1/rows.jsonl")
DEFAULT_OUTPUT_ROOT = Path("outputs/runs")

# E1A constants (4 pairs, 14 rows, k=7 rotation)
E1A_PAIRS = ["JA-EN", "KO-EN", "AR-EN", "JA-Zh"]
E1A_N_ROWS = 14
E1A_ROTATION = 7  # j = (i + 7) % 14

# E1B constants
# Primary lens: 8 content-parallel pairs (filtered from inspection of v2 packet)
E1B_CONTENT_PARALLEL = {
    ("json_format", "321"),
    ("json_format", "1148"),
    ("quotation", "122"),
    ("quotation", "281"),
    ("quotation", "2015"),
    ("no_comma", "1162"),
    ("no_comma", "1187"),
    ("no_comma", "2311"),
}
E1B_PRIMARY_ROTATION = 4  # j = (i + 4) % 8

# E1C constants
E1C_TASK_FAMILIES = ("ja-datapilot", "ko-law", "ko-legal-qa")


# ============================================================================
# Data structures
# ============================================================================


@dataclass
class E1ARow:
    """One side of a bidirectional E1A pair."""
    prompt_idx: int
    pair_label: str
    side: str  # "A" or "B"
    source_row_id: str
    source_lang: str
    target_lang: str
    prompt: str


@dataclass
class E1BPair:
    """A JA-KO Marco-MIF pair (one source_row_id within one subtask)."""
    pair_idx: int
    subtask: str
    source_row_id: str
    is_content_parallel: bool
    ja_prompt: str
    ko_prompt: str


@dataclass
class E1CRow:
    """A single-language enterprise row (no cross-language pair)."""
    prompt_idx: int
    packet_row_id: str
    task_family: str  # ja-datapilot / ko-law / ko-legal-qa
    source_row_id: str
    prompt: str


# ============================================================================
# Loading
# ============================================================================


def load_e1a_rows(corpus_path: Path) -> List[E1ARow]:
    raw = [json.loads(line) for line in corpus_path.read_text().splitlines() if line.strip()]
    rows: List[E1ARow] = []
    # Order: pair_label (in E1A_PAIRS order), source_row_id (0..13), side (A then B)
    pair_order = {p: i for i, p in enumerate(E1A_PAIRS)}
    raw_sorted = sorted(
        raw,
        key=lambda r: (
            pair_order.get(r["pair_label"], 999),
            int(r["source_row_id"]),
            r["side"],
        ),
    )
    for idx, r in enumerate(raw_sorted):
        rows.append(
            E1ARow(
                prompt_idx=idx,
                pair_label=r["pair_label"],
                side=r["side"],
                source_row_id=r["source_row_id"],
                source_lang=r["source_lang"],
                target_lang=r["target_lang"],
                prompt=r["prompt"],
            )
        )
    expected = len(E1A_PAIRS) * E1A_N_ROWS * 2
    if len(rows) != expected:
        raise SystemExit(
            f"E1A corpus has {len(rows)} rows; expected {expected} "
            f"({len(E1A_PAIRS)} pairs × {E1A_N_ROWS} rows × 2 sides)"
        )
    return rows


def load_e1b_pairs(packet_path: Path) -> List[E1BPair]:
    raw = [json.loads(line) for line in packet_path.read_text().splitlines() if line.strip()]
    ja_mif = {
        r["packet_row_id"]: r for r in raw if r["packet_row_id"].startswith("ja-mif")
    }
    ko_mif = {
        r["packet_row_id"]: r for r in raw if r["packet_row_id"].startswith("ko-mif")
    }
    pairs: List[E1BPair] = []
    for ja_id, ja_row in ja_mif.items():
        src = ja_row["source_row_id"]
        # subtask is the third dash-separated piece, e.g., "json_format" / "no_comma" / "quotation"
        subtask = ja_row["packet_row_id"].split("-")[2]
        ko_match = [
            r for r in ko_mif.values()
            if r["source_row_id"] == src and subtask in r["packet_row_id"]
        ]
        if not ko_match:
            continue
        ko_row = ko_match[0]
        pairs.append(
            E1BPair(
                pair_idx=len(pairs),
                subtask=subtask,
                source_row_id=src,
                is_content_parallel=(subtask, src) in E1B_CONTENT_PARALLEL,
                ja_prompt=ja_row["prompt"],
                ko_prompt=ko_row["prompt"],
            )
        )
    # Sort with content-parallel pairs first, then by subtask, source_row_id
    pairs.sort(
        key=lambda p: (not p.is_content_parallel, p.subtask, p.source_row_id)
    )
    # Re-index after sort
    for i, p in enumerate(pairs):
        pairs[i] = E1BPair(
            pair_idx=i, subtask=p.subtask, source_row_id=p.source_row_id,
            is_content_parallel=p.is_content_parallel,
            ja_prompt=p.ja_prompt, ko_prompt=p.ko_prompt,
        )
    if len(pairs) != 17:
        raise SystemExit(
            f"E1B paired Marco-MIF rows: expected 17, got {len(pairs)}"
        )
    n_parallel = sum(1 for p in pairs if p.is_content_parallel)
    if n_parallel != 8:
        raise SystemExit(
            f"E1B content-parallel pairs: expected 8, got {n_parallel}"
        )
    return pairs


def load_e1c_rows(packet_path: Path) -> List[E1CRow]:
    raw = [json.loads(line) for line in packet_path.read_text().splitlines() if line.strip()]
    rows: List[E1CRow] = []
    for r in raw:
        pid = r["packet_row_id"]
        if pid.startswith("ja-datapilot-"):
            family = "ja-datapilot"
        elif pid.startswith("ko-law-"):
            family = "ko-law"
        elif pid.startswith("ko-legal-qa-"):
            family = "ko-legal-qa"
        else:
            continue
        rows.append(
            E1CRow(
                prompt_idx=0,  # filled in after sort
                packet_row_id=pid,
                task_family=family,
                source_row_id=r["source_row_id"],
                prompt=r["prompt"],
            )
        )
    family_order = {f: i for i, f in enumerate(E1C_TASK_FAMILIES)}
    rows.sort(key=lambda r: (family_order[r.task_family], r.source_row_id))
    rows = [
        E1CRow(
            prompt_idx=i,
            packet_row_id=r.packet_row_id,
            task_family=r.task_family,
            source_row_id=r.source_row_id,
            prompt=r.prompt,
        )
        for i, r in enumerate(rows)
    ]
    if len(rows) != 36:
        raise SystemExit(
            f"E1C unpaired enterprise rows: expected 36 (12 ja-datapilot + 12 ko-law "
            f"+ 12 ko-legal-qa), got {len(rows)}"
        )
    return rows


# ============================================================================
# Chat template (passthrough convention from v2)
# ============================================================================


def format_chat_prompt(tokenizer: Any, user_prompt: str) -> str:
    """Render through chat template if available; otherwise passthrough.

    Tiny Aya tokenizer has chat_template=None on all snapshots, so this is a
    passthrough. Matches src/dataset_experiments.py:75 convention.
    """
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return user_prompt


# ============================================================================
# Hidden-state extraction
# ============================================================================


def last_token_hidden_states(
    model: Any,
    tokenizer: Any,
    formatted_prompt: str,
    device: Any,
) -> np.ndarray:
    """Run a forward pass and return last-position residuals at every layer.

    Returns an array of shape (n_layers + 1, hidden_size), float32.
    """
    import torch

    input_ids = tokenizer.encode(
        formatted_prompt, return_tensors="pt", add_special_tokens=False
    ).to(device)
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True, use_cache=False)
    states = out.hidden_states  # tuple of length n_layers + 1
    last = np.stack(
        [
            h[0, -1, :].detach().to("cpu", dtype=torch.float32).numpy()
            for h in states
        ],
        axis=0,
    )
    return last


# ============================================================================
# Cosine + bootstrap
# ============================================================================


def cosine_along_hidden(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine over the last dim. a, b: (..., hidden) -> (...)."""
    num = (a * b).sum(axis=-1)
    denom = (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)) + 1e-12
    return num / denom


def bootstrap_mean_ci(
    values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0,
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return (math.nan, math.nan, math.nan)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = values[idx].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


# ============================================================================
# Mode: render-only
# ============================================================================


def render_only(args: argparse.Namespace) -> None:
    """CPU sanity: load tokenizer, render E1A and E1B prompts, assert anchor."""
    from transformers import AutoTokenizer

    if not args.tokenizer_path and not args.model_id:
        raise SystemExit("--render-only needs --tokenizer-path or --model-id")
    tok_path = args.tokenizer_path or args.model_id
    LOGGER.info("Loading tokenizer from %s", tok_path)
    tokenizer = AutoTokenizer.from_pretrained(tok_path)

    # E1A: render one row per pair (row 0).
    e1a_rows = load_e1a_rows(args.triang_corpus)
    print("\n" + "=" * 80)
    print("E1A ANCHOR CHECK — first source_row_id of each pair, both sides")
    print("=" * 80)
    last_token_ids = []
    for pair in E1A_PAIRS:
        for side in ("A", "B"):
            match = [
                r for r in e1a_rows
                if r.pair_label == pair and r.side == side and r.source_row_id == "0"
            ]
            if not match:
                continue
            r = match[0]
            formatted = format_chat_prompt(tokenizer, r.prompt)
            ids = tokenizer.encode(formatted, add_special_tokens=False)
            last_token_ids.append(ids[-1])
            print(
                f"\n--- {pair} side {side} ({r.source_lang}->{r.target_lang}) ---"
            )
            print(f"  prompt last 80 chars: {r.prompt[-80:]!r}")
            print(f"  n_tokens={len(ids)}, last_id={ids[-1]} text={tokenizer.decode([ids[-1]])!r}")
    distinct = set(last_token_ids)
    print(
        f"\nAnchor check (E1A): distinct last-token ids across {len(last_token_ids)} variants = {distinct}"
    )
    if len(distinct) != 1:
        print("WARNING: E1A last-token ids are not all identical.")
    else:
        print("PASS (E1A): all variants share a single last-token id.")

    # E1B: render the 8 content-parallel pairs.
    e1b_pairs = load_e1b_pairs(args.packet_path)
    parallel = [p for p in e1b_pairs if p.is_content_parallel]
    print("\n" + "=" * 80)
    print(f"E1B ANCHOR CHECK — {len(parallel)} content-parallel JA-KO Marco-MIF pairs")
    print("=" * 80)
    last_ids_e1b = []
    for p in parallel:
        for which, prompt in (("JA", p.ja_prompt), ("KO", p.ko_prompt)):
            formatted = format_chat_prompt(tokenizer, prompt)
            ids = tokenizer.encode(formatted, add_special_tokens=False)
            last_ids_e1b.append(ids[-1])
            print(
                f"\n[{p.subtask}/{p.source_row_id}/{which}] last_id={ids[-1]} "
                f"text={tokenizer.decode([ids[-1]])!r}"
            )
            print(f"  last 80 chars: {prompt[-80:]!r}")
    distinct_b = set(last_ids_e1b)
    print(
        f"\nAnchor check (E1B): distinct last-token ids across {len(last_ids_e1b)} prompts = {distinct_b}"
    )
    if len(distinct_b) != 1:
        print(
            "NOTE (E1B): Marco-MIF prompts do not all share a single last-token id "
            "because they are real instruction-following prompts (not the "
            "translation_calibration template). The cosine is at the last position "
            "per prompt; the comparison is meaningful as long as both sides of a "
            "pair have well-defined final residuals."
        )

    # E1C: report row counts only.
    e1c_rows = load_e1c_rows(args.packet_path)
    fam_counts: Dict[str, int] = {}
    for r in e1c_rows:
        fam_counts[r.task_family] = fam_counts.get(r.task_family, 0) + 1
    print("\n" + "=" * 80)
    print("E1C ROW COUNTS (single-language enterprise tasks)")
    print("=" * 80)
    for fam, n in sorted(fam_counts.items()):
        print(f"  {fam}: {n}")


# ============================================================================
# Mode: MDE sanity (synthetic CI half-widths)
# ============================================================================


def mde_sanity(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    n_layers = args.synth_n_layers
    hidden = args.synth_hidden
    print("\n=== E1 MDE sanity (synthetic random hidden states) ===")
    print(f"hidden={hidden}, n_layers={n_layers}, n_boot=2000, seed={args.seed}\n")

    for n_pairs, k in [(8, E1B_PRIMARY_ROTATION), (14, E1A_ROTATION), (17, 8)]:
        h_a = rng.standard_normal((n_pairs, n_layers, hidden)).astype(np.float32)
        h_b = rng.standard_normal((n_pairs, n_layers, hidden)).astype(np.float32)

        def cos_along(a, b):
            num = (a * b).sum(axis=-1)
            denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12
            return num / denom

        j_idx = (np.arange(n_pairs) + k) % n_pairs
        B = cos_along(h_a, h_b)
        D = cos_along(h_a, h_b[j_idx])
        delta = B - D
        halfwidths = []
        for L in range(n_layers):
            _, lo, hi = bootstrap_mean_ci(
                delta[:, L], n_boot=2000, seed=args.seed + L
            )
            halfwidths.append((hi - lo) / 2)
        halfwidths = np.array(halfwidths)
        max_hw = float(halfwidths.max())
        median_hw = float(np.median(halfwidths))
        threshold = 0.1
        resolvable = max_hw <= threshold / 2
        verdict = "RESOLVABLE" if resolvable else "AT-RISK; consider lowering threshold"
        print(
            f"  N={n_pairs:>2}, k={k:>2}: median half-width={median_hw:.4f}, "
            f"max half-width={max_hw:.4f}; "
            f"{threshold}-threshold {'is' if resolvable else 'is NOT'} 2x resolvable -> {verdict}"
        )

    print(
        "\nE1A (N=14) + E1B primary (N=8) + E1B secondary (N=17) all share the "
        "0.1 threshold per the pre-registration. Lower if any size is AT-RISK."
    )


# ============================================================================
# Mode: extract (GPU)
# ============================================================================


def run_extract(args: argparse.Namespace) -> None:
    """Forward-pass extraction for ONE model. Saves hidden states + CSVs."""
    from src.logit_lens import load_model_and_tokenizer  # type: ignore[import-not-found]

    if not args.model_id or not args.model_slug:
        raise SystemExit("Forward-pass extraction needs --model-id and --model-slug")

    out_root = args.output_root / args.run_tag / "experiment_e1" / args.model_slug
    out_root.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Output root: %s", out_root)

    e1a_rows = load_e1a_rows(args.triang_corpus)
    e1b_pairs = load_e1b_pairs(args.packet_path)
    e1c_rows = load_e1c_rows(args.packet_path)

    LOGGER.info(
        "Loaded prompts: E1A=%d, E1B paired pairs=%d (16 prompts/pair structure: ja+ko), E1C=%d",
        len(e1a_rows), len(e1b_pairs), len(e1c_rows),
    )

    LOGGER.info("Loading model %s on %s ...", args.model_id, args.device)
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(args.model_id, device=args.device)
    LOGGER.info("Model loaded in %.1fs", time.time() - t0)
    device = next(model.parameters()).device

    # E1A forward pass.
    LOGGER.info("E1A forward pass: %d prompts ...", len(e1a_rows))
    e1a_states = np.zeros((len(e1a_rows), 37, 2048), dtype=np.float32)
    for r in e1a_rows:
        formatted = format_chat_prompt(tokenizer, r.prompt)
        e1a_states[r.prompt_idx] = last_token_hidden_states(
            model, tokenizer, formatted, device
        )
        if (r.prompt_idx + 1) % 16 == 0:
            LOGGER.info("  E1A %d/%d", r.prompt_idx + 1, len(e1a_rows))
    np.save(out_root / "e1a_hidden_states.npy", e1a_states)
    pd.DataFrame([
        {
            "prompt_idx": r.prompt_idx,
            "pair_label": r.pair_label,
            "side": r.side,
            "source_row_id": r.source_row_id,
            "source_lang": r.source_lang,
            "target_lang": r.target_lang,
        }
        for r in e1a_rows
    ]).to_csv(out_root / "e1a_metadata.csv", index=False)

    # E1A within-model cosines.
    e1a_cosine_long, e1a_delta = compute_e1a_cosines(e1a_rows, e1a_states, args.model_slug)
    e1a_cosine_long.to_csv(out_root / "e1a_per_layer_cosine.csv", index=False)
    e1a_delta.to_csv(out_root / "e1a_delta_meaning_summary.csv", index=False)

    # E1B forward pass (each paired row has 2 prompts: JA and KO).
    LOGGER.info("E1B forward pass: %d prompts ...", len(e1b_pairs) * 2)
    e1b_states = np.zeros((len(e1b_pairs) * 2, 37, 2048), dtype=np.float32)
    e1b_meta_rows: List[Dict[str, Any]] = []
    for p in e1b_pairs:
        ja_idx = 2 * p.pair_idx
        ko_idx = 2 * p.pair_idx + 1
        ja_fmt = format_chat_prompt(tokenizer, p.ja_prompt)
        ko_fmt = format_chat_prompt(tokenizer, p.ko_prompt)
        e1b_states[ja_idx] = last_token_hidden_states(model, tokenizer, ja_fmt, device)
        e1b_states[ko_idx] = last_token_hidden_states(model, tokenizer, ko_fmt, device)
        e1b_meta_rows.extend([
            {
                "prompt_idx": ja_idx, "pair_idx": p.pair_idx, "side": "JA",
                "subtask": p.subtask, "source_row_id": p.source_row_id,
                "is_content_parallel": p.is_content_parallel,
            },
            {
                "prompt_idx": ko_idx, "pair_idx": p.pair_idx, "side": "KO",
                "subtask": p.subtask, "source_row_id": p.source_row_id,
                "is_content_parallel": p.is_content_parallel,
            },
        ])
    np.save(out_root / "e1b_hidden_states.npy", e1b_states)
    pd.DataFrame(e1b_meta_rows).to_csv(out_root / "e1b_metadata.csv", index=False)

    # E1B within-model cosines (primary 8 + secondary 17).
    e1b_cosine_long, e1b_delta = compute_e1b_cosines(e1b_pairs, e1b_states, args.model_slug)
    e1b_cosine_long.to_csv(out_root / "e1b_per_layer_cosine.csv", index=False)
    e1b_delta.to_csv(out_root / "e1b_delta_meaning_summary.csv", index=False)

    # E1C forward pass.
    LOGGER.info("E1C forward pass: %d prompts ...", len(e1c_rows))
    e1c_states = np.zeros((len(e1c_rows), 37, 2048), dtype=np.float32)
    for r in e1c_rows:
        formatted = format_chat_prompt(tokenizer, r.prompt)
        e1c_states[r.prompt_idx] = last_token_hidden_states(
            model, tokenizer, formatted, device
        )
        if (r.prompt_idx + 1) % 12 == 0:
            LOGGER.info("  E1C %d/%d", r.prompt_idx + 1, len(e1c_rows))
    np.save(out_root / "e1c_hidden_states.npy", e1c_states)
    pd.DataFrame([
        {
            "prompt_idx": r.prompt_idx, "packet_row_id": r.packet_row_id,
            "task_family": r.task_family, "source_row_id": r.source_row_id,
        }
        for r in e1c_rows
    ]).to_csv(out_root / "e1c_metadata.csv", index=False)

    LOGGER.info("Extraction complete for %s. Run --analyze after all 3 models extracted.", args.model_slug)


# ============================================================================
# Within-model cosines: E1A
# ============================================================================


def compute_e1a_cosines(
    rows: List[E1ARow], states: np.ndarray, model_slug: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute B/C/D/E cosines per pair_label, then bootstrap CI on B-D."""
    n_layers = states.shape[1]

    # Group: (pair_label) -> (sideA states (14,L,H), sideB states (14,L,H))
    long_rows: List[Dict[str, Any]] = []
    delta_rows: List[Dict[str, Any]] = []
    for pair in E1A_PAIRS:
        side_a_idx = sorted([r.prompt_idx for r in rows if r.pair_label == pair and r.side == "A"],
                            key=lambda i: int(rows[i].source_row_id))
        side_b_idx = sorted([r.prompt_idx for r in rows if r.pair_label == pair and r.side == "B"],
                            key=lambda i: int(rows[i].source_row_id))
        if len(side_a_idx) != E1A_N_ROWS or len(side_b_idx) != E1A_N_ROWS:
            raise SystemExit(f"E1A: pair {pair} has wrong row count")
        H_A = states[side_a_idx]  # (14, 37, 2048)
        H_B = states[side_b_idx]

        n_pairs = E1A_N_ROWS
        j_idx = [(i + E1A_ROTATION) % n_pairs for i in range(n_pairs)]

        B = cosine_along_hidden(H_A, H_B)               # (14, 37)
        C = cosine_along_hidden(H_A, H_A[j_idx])
        D = cosine_along_hidden(H_A, H_B[j_idx])
        E = cosine_along_hidden(H_B, H_B[j_idx])

        for cond_name, cos in [
            ("B_same_meaning_cross_dir", B),
            ("C_diff_meaning_same_dir_A", C),
            ("D_diff_meaning_cross_dir", D),
            ("E_diff_meaning_same_dir_B", E),
        ]:
            for i in range(n_pairs):
                for L in range(n_layers):
                    long_rows.append({
                        "model_slug": model_slug,
                        "pair_label": pair,
                        "pair_i": i,
                        "source_row_id_i": rows[side_a_idx[i]].source_row_id,
                        "source_row_id_j": rows[side_a_idx[j_idx[i]]].source_row_id,
                        "layer": L,
                        "condition": cond_name,
                        "cosine": float(cos[i, L]),
                    })

        # Delta (B - D) per layer with bootstrap CI.
        delta = B - D  # (14, 37)
        for L in range(n_layers):
            mean, lo, hi = bootstrap_mean_ci(delta[:, L], n_boot=2000, seed=int(L) * 13 + 5)
            in_window = 10 <= L <= 35
            delta_rows.append({
                "model_slug": model_slug,
                "pair_label": pair,
                "layer": L,
                "delta_mean": mean,
                "delta_ci_lo": lo,
                "delta_ci_hi": hi,
                "delta_ci_halfwidth": (hi - lo) / 2,
                "in_gate_window": in_window,
                "gate_pass_layer": bool(in_window and lo > 0.0),
            })

    return pd.DataFrame(long_rows), pd.DataFrame(delta_rows)


# ============================================================================
# Within-model cosines: E1B
# ============================================================================


def compute_e1b_cosines(
    pairs: List[E1BPair], states: np.ndarray, model_slug: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute B-D cosines for primary lens (8 parallel) and secondary lens (17 all)."""
    n_layers = states.shape[1]
    long_rows: List[Dict[str, Any]] = []
    delta_rows: List[Dict[str, Any]] = []

    for lens_name, lens_filter, rotation in [
        ("primary_8_content_parallel", lambda p: p.is_content_parallel, E1B_PRIMARY_ROTATION),
        ("secondary_17_all_paired", lambda p: True, 8),
    ]:
        lens_pairs = [p for p in pairs if lens_filter(p)]
        if not lens_pairs:
            continue
        n_pairs = len(lens_pairs)
        ja_idx = [2 * p.pair_idx for p in lens_pairs]
        ko_idx = [2 * p.pair_idx + 1 for p in lens_pairs]
        H_JA = states[ja_idx]  # (n_pairs, 37, 2048)
        H_KO = states[ko_idx]

        j_idx = [(i + rotation) % n_pairs for i in range(n_pairs)]
        B = cosine_along_hidden(H_JA, H_KO)
        D = cosine_along_hidden(H_JA, H_KO[j_idx])

        for cond_name, cos in [
            ("B_same_meaning_cross_lang", B),
            ("D_diff_meaning_cross_lang", D),
        ]:
            for i in range(n_pairs):
                for L in range(n_layers):
                    long_rows.append({
                        "model_slug": model_slug,
                        "lens": lens_name,
                        "pair_i": i,
                        "subtask": lens_pairs[i].subtask,
                        "source_row_id": lens_pairs[i].source_row_id,
                        "layer": L,
                        "condition": cond_name,
                        "cosine": float(cos[i, L]),
                    })

        delta = B - D
        for L in range(n_layers):
            mean, lo, hi = bootstrap_mean_ci(delta[:, L], n_boot=2000, seed=int(L) * 17 + 11)
            in_window = 10 <= L <= 35
            delta_rows.append({
                "model_slug": model_slug,
                "lens": lens_name,
                "n_pairs": n_pairs,
                "rotation_k": rotation,
                "layer": L,
                "delta_mean": mean,
                "delta_ci_lo": lo,
                "delta_ci_hi": hi,
                "delta_ci_halfwidth": (hi - lo) / 2,
                "in_gate_window": in_window,
                "gate_pass_layer": bool(in_window and lo > 0.0),
            })

    return pd.DataFrame(long_rows), pd.DataFrame(delta_rows)


# ============================================================================
# Mode: analyze (cross-model post-processing for E1C)
# ============================================================================


def run_analyze(args: argparse.Namespace) -> None:
    """Read hidden states from all models; compute E1C cross-checkpoint cosines."""
    out_root = args.output_root / args.run_tag / "experiment_e1"
    if not out_root.exists():
        raise SystemExit(f"Run directory not found: {out_root}")

    # Discover model slugs.
    slugs = sorted([p.name for p in out_root.iterdir() if p.is_dir()])
    if not slugs:
        raise SystemExit(f"No model subdirectories found under {out_root}")
    LOGGER.info("Found model slugs: %s", slugs)

    base_slug = next((s for s in slugs if "base" in s.lower()), None)
    if not base_slug:
        raise SystemExit(
            f"E1C analyze needs a Base model slug containing 'base'; found: {slugs}"
        )
    post_slugs = [s for s in slugs if s != base_slug]
    LOGGER.info("Base: %s; post-trained: %s", base_slug, post_slugs)

    # Load E1C hidden states + metadata.
    base_states = np.load(out_root / base_slug / "e1c_hidden_states.npy")  # (36, 37, 2048)
    base_meta = pd.read_csv(out_root / base_slug / "e1c_metadata.csv")

    e1c_rows: List[Dict[str, Any]] = []
    for post_slug in post_slugs:
        post_states = np.load(out_root / post_slug / "e1c_hidden_states.npy")
        # cos along last dim per (row, layer).
        cos = cosine_along_hidden(base_states, post_states)  # (36, 37)
        for i in range(cos.shape[0]):
            row_meta = base_meta.iloc[i]
            for L in range(cos.shape[1]):
                e1c_rows.append({
                    "post_slug": post_slug,
                    "task_family": row_meta["task_family"],
                    "packet_row_id": row_meta["packet_row_id"],
                    "source_row_id": row_meta["source_row_id"],
                    "layer": L,
                    "cross_checkpoint_cosine": float(cos[i, L]),
                })

    e1c_long = pd.DataFrame(e1c_rows)
    e1c_long.to_csv(out_root / "e1c_cross_checkpoint_cosine.csv", index=False)
    LOGGER.info("Wrote %s", out_root / "e1c_cross_checkpoint_cosine.csv")

    # Aggregate per-task per-layer per-post-checkpoint mean + bootstrap CI.
    summary_rows: List[Dict[str, Any]] = []
    for (post_slug, task_fam, layer), group in e1c_long.groupby(
        ["post_slug", "task_family", "layer"]
    ):
        vals = group["cross_checkpoint_cosine"].to_numpy()
        mean, lo, hi = bootstrap_mean_ci(vals, n_boot=2000, seed=int(layer) * 19 + 23)
        summary_rows.append({
            "post_slug": post_slug,
            "task_family": task_fam,
            "layer": int(layer),
            "n_rows": int(len(vals)),
            "mean": mean,
            "ci_lo": lo,
            "ci_hi": hi,
            "ci_halfwidth": (hi - lo) / 2,
        })
    e1c_summary = pd.DataFrame(summary_rows)
    e1c_summary.to_csv(out_root / "e1c_cross_checkpoint_summary.csv", index=False)
    LOGGER.info("Wrote %s", out_root / "e1c_cross_checkpoint_summary.csv")

    # Identify minimum layer per (post_slug, task_family) in [10, 35].
    min_rows: List[Dict[str, Any]] = []
    for (post_slug, task_fam), group in e1c_summary.groupby(["post_slug", "task_family"]):
        in_window = group[(group["layer"] >= 10) & (group["layer"] <= 35)]
        if in_window.empty:
            continue
        idx = in_window["mean"].idxmin()
        row = e1c_summary.loc[idx]
        min_rows.append({
            "post_slug": post_slug,
            "task_family": task_fam,
            "min_layer": int(row["layer"]),
            "min_mean": row["mean"],
            "min_ci_lo": row["ci_lo"],
            "min_ci_hi": row["ci_hi"],
            "in_route_window_24_34": bool(24 <= int(row["layer"]) <= 34),
        })
    e1c_min = pd.DataFrame(min_rows)
    e1c_min.to_csv(out_root / "e1c_min_layer_summary.csv", index=False)
    print("\n=== E1C cross-checkpoint minimum layer per (post_slug, task_family) ===")
    print(e1c_min.to_string(index=False))

    # Aggregate E1A and E1B verdicts across models.
    e1a_summary_rows: List[Dict[str, Any]] = []
    e1b_summary_rows: List[Dict[str, Any]] = []
    for slug in slugs:
        e1a_path = out_root / slug / "e1a_delta_meaning_summary.csv"
        if e1a_path.exists():
            df = pd.read_csv(e1a_path)
            for pair_label, group in df.groupby("pair_label"):
                in_window = group[
                    (group["layer"] >= 10) & (group["layer"] <= 35)
                ]
                if in_window.empty:
                    continue
                idx = in_window["delta_mean"].idxmax()
                row = df.loc[idx]
                e1a_summary_rows.append({
                    "model_slug": slug,
                    "pair_label": pair_label,
                    "peak_layer": int(row["layer"]),
                    "peak_delta_mean": row["delta_mean"],
                    "peak_ci_lo": row["delta_ci_lo"],
                    "peak_ci_hi": row["delta_ci_hi"],
                    "peak_in_24_34": bool(24 <= int(row["layer"]) <= 34),
                    "peak_above_0_1_with_lower_ci_pos": bool(
                        row["delta_mean"] > 0.1 and row["delta_ci_lo"] > 0
                    ),
                })

        e1b_path = out_root / slug / "e1b_delta_meaning_summary.csv"
        if e1b_path.exists():
            df = pd.read_csv(e1b_path)
            for (lens, _), group in df.groupby(["lens", "n_pairs"]):
                in_window = group[
                    (group["layer"] >= 10) & (group["layer"] <= 35)
                ]
                if in_window.empty:
                    continue
                idx = in_window["delta_mean"].idxmax()
                row = df.loc[idx]
                e1b_summary_rows.append({
                    "model_slug": slug,
                    "lens": lens,
                    "n_pairs": int(row["n_pairs"]),
                    "peak_layer": int(row["layer"]),
                    "peak_delta_mean": row["delta_mean"],
                    "peak_ci_lo": row["delta_ci_lo"],
                    "peak_ci_hi": row["delta_ci_hi"],
                    "peak_in_24_34": bool(24 <= int(row["layer"]) <= 34),
                    "peak_above_0_1_with_lower_ci_pos": bool(
                        row["delta_mean"] > 0.1 and row["delta_ci_lo"] > 0
                    ),
                })

    if e1a_summary_rows:
        df_e1a = pd.DataFrame(e1a_summary_rows)
        df_e1a.to_csv(out_root / "e1a_peak_layer_by_model_pair.csv", index=False)
        print("\n=== E1A peak (B - D) layer per (model, pair) ===")
        print(df_e1a.to_string(index=False))
    if e1b_summary_rows:
        df_e1b = pd.DataFrame(e1b_summary_rows)
        df_e1b.to_csv(out_root / "e1b_peak_layer_by_model_lens.csv", index=False)
        print("\n=== E1B peak (B - D) layer per (model, lens) ===")
        print(df_e1b.to_string(index=False))


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=None,
                        help="HF id or local path; required for --extract / --analyze (e.g., CohereLabs/tiny-aya-base)")
    parser.add_argument("--model-slug", default=None,
                        help="Short slug for output dir; required for extract")
    parser.add_argument("--triang-corpus", type=Path, default=DEFAULT_TRIANG_CORPUS)
    parser.add_argument("--packet-path", type=Path, default=None,
                        help="JSONL path; required for --extract / --analyze (schema in README)")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-tag", default=None,
                        help="Subfolder under output-root; defaults to timestamp")
    parser.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    parser.add_argument("--render-only", action="store_true",
                        help="CPU tokenizer-only sanity")
    parser.add_argument("--mde-sanity", action="store_true",
                        help="Synthetic MDE check at N in {8, 14, 17}")
    parser.add_argument("--analyze", action="store_true",
                        help="Cross-model post-processing for E1C")
    parser.add_argument("--tokenizer-path", default=None,
                        help="Tokenizer path/id for --render-only; defaults to --model-id")
    parser.add_argument("--synth-n-layers", type=int, default=37)
    parser.add_argument("--synth-hidden", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    if args.mde_sanity:
        mde_sanity(args)
        return
    if args.render_only:
        render_only(args)
        return
    if args.analyze:
        if not args.run_tag:
            raise SystemExit("--analyze requires --run-tag (the run directory to analyze)")
        run_analyze(args)
        return

    if not args.run_tag:
        args.run_tag = time.strftime("%Y%m%d-%H%M%S")
    run_extract(args)


if __name__ == "__main__":
    main()
