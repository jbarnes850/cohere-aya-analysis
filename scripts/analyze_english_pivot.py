#!/usr/bin/env python3
"""Experiment E2b: English-pivot logit-lens test on E1A hidden states.

Tests the falsifiable claim from the E2 pre-reg:
  Logit-lens projection of h[L] @ W_unembed.T at L in [20, 35] across the
  4 E1A pairs shows that English-target prompts (Side A: X→EN) have
  monotonically climbing English-token mass from L=20 through L=30, while
  non-English-target prompts (Side B: EN→X) show separation from English-
  token mass after L=24.

The test is grounded in Anthropic's "Biology of an LLM" (2025) finding that
multilingual features have direct weights to English output nodes; non-English
outputs are mediated by say-X-in-language-Y features.

Inputs:
  - E1A hidden states from <output-root>/<run-tag>/experiment_e1/
    <slug>/e1a_hidden_states.npy and e1a_metadata.csv
  - W_unembed (lm_head.weight) from each model — loaded once per model

Outputs:
  <output-root>/<e1-run-tag>/experiment_e2_english_pivot/
    per_layer_lang_token_mass.csv  long-format: model_slug, pair_label, side,
                                    source_row_id, layer, lang_script, mass
    english_pivot_summary.csv      aggregated mean per (model, pair, side, layer, lang)
    english_pivot_verdict.md       text verdict against the predictions

For each (model, pair, side, source_row_id, layer):
  - Project hidden state h[L, :] through W_unembed.T to get token logits
  - Apply softmax to get token probabilities
  - Group tokens by Unicode script category (Latin, Han, Hangul, Arabic, ...)
  - Sum probability mass per script
  - Identify "English-token-mass" as Latn-script tokens

Modes:
  --extract-w-unembed   Load model and save W_unembed to a numpy file (run once per model).
  --analyze             CPU-only: load saved hidden states + W_unembed, compute mass, write CSVs.
  --smoke               Like --analyze but on synthetic data; verifies pipeline.
"""
from __future__ import annotations

import argparse
import logging
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Make src.* importable.
_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))


LOGGER = logging.getLogger("e2_english_pivot")

DEFAULT_OUTPUT_ROOT = Path("outputs/runs")

LAYERS_OF_INTEREST = list(range(20, 36))  # L=20..35 inclusive


def script_of_codepoint(cp: int) -> str:
    """Map a codepoint to a coarse script label."""
    try:
        c = chr(cp)
    except (ValueError, OverflowError):
        return "OTHER"
    name = unicodedata.name(c, "")
    if not name:
        return "OTHER"
    # Coarse buckets aligned with our 5-language target set.
    if "LATIN" in name:
        return "Latn"
    if "CJK UNIFIED IDEOGRAPH" in name or "CJK COMPATIBILITY" in name:
        return "Hani"
    if "HIRAGANA" in name or "KATAKANA" in name:
        return "Jpan"  # Japanese kana (kanji counted as Hani)
    if "HANGUL" in name:
        return "Hang"
    if "ARABIC" in name:
        return "Arab"
    if "GREEK" in name:
        return "Grek"
    if "CYRILLIC" in name:
        return "Cyrl"
    if "DEVANAGARI" in name:
        return "Deva"
    return "OTHER"


def build_token_to_script_map(tokenizer: Any) -> np.ndarray:
    """Return an (vocab_size,) array of script labels for each token id."""
    vocab_size = len(tokenizer)
    LOGGER.info("Building token->script map for vocab_size=%d", vocab_size)
    scripts = np.full(vocab_size, "OTHER", dtype=object)
    for tid in range(vocab_size):
        try:
            decoded = tokenizer.decode([tid])
        except Exception:
            scripts[tid] = "OTHER"
            continue
        if not decoded:
            scripts[tid] = "OTHER"
            continue
        # Use the most common script across the decoded characters.
        cs = [script_of_codepoint(ord(ch)) for ch in decoded]
        # Drop OTHER if there's a "real" script in the token.
        real = [s for s in cs if s != "OTHER"]
        if real:
            scripts[tid] = max(set(real), key=real.count)
        else:
            scripts[tid] = "OTHER"
    return scripts


def extract_w_unembed(args: argparse.Namespace) -> None:
    """Load model and save W_unembed (vocab × hidden) to a numpy file."""
    from src.logit_lens import load_model_and_tokenizer  # type: ignore[import-not-found]

    if not args.model_id or not args.model_slug:
        raise SystemExit("--extract-w-unembed needs --model-id and --model-slug")

    out_root = args.output_root / args.run_tag / "experiment_e2_english_pivot"
    out_root.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading model %s on %s ...", args.model_id, args.device)
    model, tokenizer = load_model_and_tokenizer(args.model_id, device=args.device)
    LOGGER.info("Model loaded.")

    # lm_head.weight is (vocab_size, hidden_size). For Tiny Aya / Cohere2 it may be tied.
    if hasattr(model, "lm_head"):
        w_unembed = model.lm_head.weight.detach().to("cpu", dtype=__import__("torch").float32).numpy()
    elif hasattr(model, "get_output_embeddings"):
        w_unembed = (
            model.get_output_embeddings().weight.detach().to("cpu", dtype=__import__("torch").float32).numpy()
        )
    else:
        raise SystemExit("Could not locate lm_head / output embeddings on model.")
    LOGGER.info("W_unembed shape: %s", w_unembed.shape)

    np.save(out_root / f"w_unembed_{args.model_slug}.npy", w_unembed)
    LOGGER.info("Saved W_unembed to %s", out_root / f"w_unembed_{args.model_slug}.npy")

    # Token->script map (use this model's tokenizer).
    script_arr = build_token_to_script_map(tokenizer)
    np.save(out_root / f"token_scripts_{args.model_slug}.npy", script_arr)
    LOGGER.info(
        "Saved token->script map (size=%d) to %s",
        len(script_arr), out_root / f"token_scripts_{args.model_slug}.npy",
    )


def softmax_logits(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over last axis."""
    m = logits.max(axis=-1, keepdims=True)
    ex = np.exp(logits - m)
    return ex / ex.sum(axis=-1, keepdims=True)


def compute_lang_mass(
    h: np.ndarray, w_unembed: np.ndarray, scripts: np.ndarray, layers: List[int],
) -> Dict[str, np.ndarray]:
    """For hidden states h (n_prompts, n_layers, hidden), W_unembed (vocab, hidden),
    return per-script probability mass at each (prompt, layer)."""
    out: Dict[str, np.ndarray] = {}
    n_prompts = h.shape[0]
    unique_scripts = sorted(set(scripts.tolist()))
    for s in unique_scripts:
        out[s] = np.zeros((n_prompts, len(layers)), dtype=np.float32)
    for li, layer in enumerate(layers):
        # h_L: (n_prompts, hidden)
        h_L = h[:, layer, :]
        # logits: (n_prompts, vocab)
        logits = h_L @ w_unembed.T  # (n_prompts, vocab)
        probs = softmax_logits(logits.astype(np.float64)).astype(np.float32)  # higher precision in softmax
        for s in unique_scripts:
            mask = scripts == s  # (vocab,)
            mass = probs[:, mask].sum(axis=-1)  # (n_prompts,)
            out[s][:, li] = mass
    return out


def run_analyze(args: argparse.Namespace) -> None:
    """Compute per-layer language-script mass for each model × pair × side."""
    out_root = args.output_root / args.run_tag / "experiment_e2_english_pivot"
    e1_root = args.output_root / args.run_tag / "experiment_e1"
    if not e1_root.exists():
        raise SystemExit(f"E1 run directory not found: {e1_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    slugs = sorted([p.name for p in e1_root.iterdir() if p.is_dir()])
    LOGGER.info("Found model slugs: %s", slugs)

    long_rows: List[Dict[str, Any]] = []
    for slug in slugs:
        w_path = out_root / f"w_unembed_{slug}.npy"
        s_path = out_root / f"token_scripts_{slug}.npy"
        if not w_path.exists() or not s_path.exists():
            LOGGER.warning(
                "Missing W_unembed or token-script map for %s. Run --extract-w-unembed first.",
                slug,
            )
            continue
        w_unembed = np.load(w_path)
        scripts = np.load(s_path, allow_pickle=True)
        e1a_h = np.load(e1_root / slug / "e1a_hidden_states.npy")
        e1a_meta = pd.read_csv(e1_root / slug / "e1a_metadata.csv")
        LOGGER.info(
            "%s: hidden_states %s, W_unembed %s, scripts vocab %d",
            slug, e1a_h.shape, w_unembed.shape, len(scripts),
        )

        mass_by_script = compute_lang_mass(e1a_h, w_unembed, scripts, LAYERS_OF_INTEREST)
        for prompt_idx in range(e1a_h.shape[0]):
            row_meta = e1a_meta.iloc[prompt_idx]
            for li, layer in enumerate(LAYERS_OF_INTEREST):
                for s, arr in mass_by_script.items():
                    long_rows.append({
                        "model_slug": slug,
                        "pair_label": row_meta["pair_label"],
                        "side": row_meta["side"],
                        "source_row_id": row_meta["source_row_id"],
                        "source_lang": row_meta["source_lang"],
                        "target_lang": row_meta["target_lang"],
                        "layer": layer,
                        "lang_script": s,
                        "mass": float(arr[prompt_idx, li]),
                    })

    df = pd.DataFrame(long_rows)
    df.to_csv(out_root / "per_layer_lang_token_mass.csv", index=False)
    LOGGER.info("Wrote %s (%d rows)", out_root / "per_layer_lang_token_mass.csv", len(df))

    # Aggregate per (model, pair, side, layer, lang).
    summary = (
        df.groupby(["model_slug", "pair_label", "side", "layer", "lang_script"])["mass"]
        .agg(["mean", "std", "count"]).reset_index()
    )
    summary.to_csv(out_root / "english_pivot_summary.csv", index=False)
    LOGGER.info("Wrote %s", out_root / "english_pivot_summary.csv")

    # Verdict text.
    verdict = compute_verdict(summary)
    (out_root / "english_pivot_verdict.md").write_text(verdict)
    LOGGER.info("Wrote %s", out_root / "english_pivot_verdict.md")
    print("\n" + verdict)


def compute_verdict(summary: pd.DataFrame) -> str:
    """Format a markdown verdict for the English-pivot test."""
    parts: List[str] = ["# E2b English-pivot test verdict\n"]
    parts.append(
        "Hypothesis: Side A (X→EN) shows monotonically climbing Latn-script "
        "token mass from L=20 to L=30; Side B (EN→X) trajectory diverges "
        "from Side A after L=24."
    )
    parts.append(
        "\nReports per-(model, pair, side) the Latn token-mass at L=20 and L=30, "
        "the climb ratio (L30/L20), and whether Side A's L30 mass exceeds Side B's L30 by 2x.\n"
    )

    summary_latn = summary[summary["lang_script"] == "Latn"].copy()
    rows: List[Dict[str, Any]] = []
    for (model, pair), g in summary_latn.groupby(["model_slug", "pair_label"]):
        side_a = g[g["side"] == "A"].set_index("layer")
        side_b = g[g["side"] == "B"].set_index("layer")
        if 20 not in side_a.index or 30 not in side_a.index or 20 not in side_b.index or 30 not in side_b.index:
            continue
        a_l20 = side_a.loc[20, "mean"]
        a_l30 = side_a.loc[30, "mean"]
        b_l20 = side_b.loc[20, "mean"]
        b_l30 = side_b.loc[30, "mean"]
        a_climb = a_l30 / max(a_l20, 1e-8)
        b_climb = b_l30 / max(b_l20, 1e-8)
        diverge = a_l30 / max(b_l30, 1e-8)
        rows.append({
            "model": model, "pair": pair,
            "A_L20_mass": a_l20, "A_L30_mass": a_l30, "A_climb": a_climb,
            "B_L20_mass": b_l20, "B_L30_mass": b_l30, "B_climb": b_climb,
            "A_over_B_at_L30": diverge,
            "A_climbs_2x": a_climb >= 2.0,
            "A_2x_B_at_L30": diverge >= 2.0,
        })
    if not rows:
        parts.append("\nNo data found for verdict.\n")
        return "\n".join(parts)
    parts.append("\n## Per-(model, pair) Latn mass climb and Side A vs Side B at L=30\n")
    parts.append("| model | pair | A_L20 | A_L30 | A_climb | B_L20 | B_L30 | B_climb | A/B@L30 | A_climbs_2x | A_2x_B |")
    parts.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        parts.append(
            f"| {r['model']} | {r['pair']} | {r['A_L20_mass']:.3f} | {r['A_L30_mass']:.3f} | "
            f"{r['A_climb']:.2f}x | {r['B_L20_mass']:.3f} | {r['B_L30_mass']:.3f} | "
            f"{r['B_climb']:.2f}x | {r['A_over_B_at_L30']:.2f}x | {r['A_climbs_2x']} | {r['A_2x_B_at_L30']} |"
        )
    n_total = len(rows)
    n_climb = sum(1 for r in rows if r["A_climbs_2x"])
    n_diverge = sum(1 for r in rows if r["A_2x_B_at_L30"])
    parts.append(f"\n**Climb count:** {n_climb}/{n_total} (Side A's L30 mass >= 2x its L20 mass).")
    parts.append(f"**Divergence count:** {n_diverge}/{n_total} (Side A's L30 mass >= 2x Side B's L30 mass).")
    if n_climb >= 0.75 * n_total and n_diverge >= 0.75 * n_total:
        verdict = "**PASS** — Tiny Aya is English-pivot-mediated for translation. Anthropic biology hypothesis holds."
    elif n_climb >= 0.5 * n_total or n_diverge >= 0.5 * n_total:
        verdict = "**MIXED** — partial English-pivot signal. Refined claim required."
    else:
        verdict = "**FAIL** — no English-pivot signature. Tiny Aya does not route through English for these pairs."
    parts.append(f"\nVerdict: {verdict}\n")
    return "\n".join(parts)


def run_smoke(args: argparse.Namespace) -> None:
    """Synthetic-data smoke for the analysis pipeline (no model needed)."""
    rng = np.random.default_rng(args.seed)
    vocab_size = 1024
    hidden = 64
    n_prompts = 4
    n_layers = 16
    LOGGER.info(
        "Smoke: vocab=%d, hidden=%d, prompts=%d, layers=%d",
        vocab_size, hidden, n_prompts, n_layers,
    )

    h = rng.standard_normal((n_prompts, n_layers, hidden)).astype(np.float32)
    w = rng.standard_normal((vocab_size, hidden)).astype(np.float32)
    scripts = np.array(
        ["Latn"] * 200 + ["Hani"] * 200 + ["Hang"] * 200 + ["Arab"] * 200 + ["OTHER"] * (vocab_size - 800),
        dtype=object,
    )
    layers = list(range(8, 16))
    mass = compute_lang_mass(h, w, scripts, layers)
    print("\nSmoke mass-by-script shapes:")
    for s, arr in mass.items():
        print(f"  {s}: shape={arr.shape}, dtype={arr.dtype}, sum-per-prompt-per-layer "
              f"mean={arr.mean():.4f}")
    # Check that all probabilities sum to 1.0 across scripts
    total = np.zeros((n_prompts, len(layers)), dtype=np.float32)
    for s in mass.values():
        total += s
    print(f"  total mass across all scripts (should be 1.0): "
          f"min={total.min():.4f}, max={total.max():.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", default=None,
                        help="Required outside --smoke")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    parser.add_argument("--model-id", default=None,
                        help="HF id or local path; required outside --smoke (e.g., CohereLabs/tiny-aya-base)")
    parser.add_argument("--model-slug", default=None)
    parser.add_argument("--extract-w-unembed", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.smoke:
        run_smoke(args)
        return 0
    if args.extract_w_unembed:
        extract_w_unembed(args)
        return 0
    if args.analyze:
        run_analyze(args)
        return 0
    raise SystemExit("Specify one of: --extract-w-unembed, --analyze, --smoke")


if __name__ == "__main__":
    sys.exit(main())
