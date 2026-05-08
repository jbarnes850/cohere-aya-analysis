#!/usr/bin/env python3
"""Experiment E2 step 3: analyze trained SAEs and run the steering test.

Three analyses:

  1. Per-feature discrimination (E2a test (a)):
     For each (model, layer) trained SAE, encode the saved E1B JA-KO Marco-MIF
     hidden states (8 content-parallel pairs × 2 sides). Rank features by
     mean-activation-on-B-condition minus D-condition discrimination score.
     Compare ranks between Base-SAE and Global-SAE: features that rank high
     in Global-SAE but low in Base-SAE are the installation signature.

  2. Maximally-activating prompts (E2a test (b)):
     For each top-k feature in Global-SAE-L26, find the corpus prompts on
     which the feature activates most. Dump prompt text for human labeling.

  3. Steering test (E2a test (c)):
     Add weighted feature direction(s) to Base's residual at L=26 during
     forward, regenerate on the 8 paired Marco-MIF rows, measure format
     fidelity / target-language script ratio / chrF lift toward Global.

Modes:
  --discriminate           CPU: compute per-feature discrimination scores
  --max-activating         CPU: dump max-activating prompts for top features
  --steering               GPU: re-generate with feature steering
  --smoke                  CPU synthetic: verify pipeline

Usage:
  python scripts/run_experiment_e2_analyze.py \
    --run-tag e2_<timestamp> --discriminate
  python scripts/run_experiment_e2_analyze.py \
    --run-tag e2_<timestamp> --max-activating --top-k 16
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Make src.* importable.
_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))


LOGGER = logging.getLogger("e2_analyze")

DEFAULT_OUTPUT_ROOT = Path("outputs/runs")
STEERING_METRICS = [
    "is_json_valid",
    "has_quotation_wrap",
    "no_comma_compliance",
    "target_lang_script_ratio",
    "chrf_to_global",
]


def load_sae_checkpoint(ckpt_path: Path):
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    return ckpt


def encode_with_sae(ckpt: Dict[str, Any], x: np.ndarray) -> np.ndarray:
    """Apply the SAE encoder + TopK to x. Returns sparse code (n, dict_size)."""
    import torch
    enc_w = ckpt["encoder_weight"]  # (dict, hidden)
    enc_b = ckpt["encoder_bias"]    # (dict,)
    k = ckpt["k"]

    x_t = torch.tensor(x, dtype=torch.float32)
    z = torch.relu(x_t @ enc_w.T + enc_b)
    topk_vals, topk_idx = torch.topk(z, k, dim=-1)
    sparse = torch.zeros_like(z)
    sparse.scatter_(-1, topk_idx, topk_vals)
    return sparse.numpy()


def run_discriminate(args: argparse.Namespace) -> None:
    """Compare per-feature discrimination on E1B B vs D conditions for Base-SAE vs Global-SAE."""
    e2_root = args.output_root / args.run_tag / "experiment_e2_sae"
    if not e2_root.exists():
        raise SystemExit(f"E2 SAE directory not found: {e2_root}")
    e1_root = args.output_root / args.e1_run_tag / "experiment_e1"

    discrim_rows: List[Dict[str, Any]] = []
    for slug in ["tiny-aya-base", "tiny-aya-global"]:
        for layer in [26, 30]:
            sae_dir = e2_root / f"{slug}_L{layer}"
            ckpt_path = sae_dir / "sae_checkpoint.pt"
            if not ckpt_path.exists():
                LOGGER.warning("SAE checkpoint missing: %s", ckpt_path)
                continue
            ckpt = load_sae_checkpoint(ckpt_path)
            LOGGER.info(
                "%s L%d SAE: dict=%d, k=%d, hidden=%d",
                slug, layer, ckpt["dict_size"], ckpt["k"], ckpt["hidden"],
            )

            # Load E1B hidden states + metadata for this model.
            e1b_h = np.load(e1_root / slug / "e1b_hidden_states.npy")  # (34, 37, 2048)
            e1b_meta = pd.read_csv(e1_root / slug / "e1b_metadata.csv")

            # Filter to content-parallel pairs only.
            parallel_meta = e1b_meta[e1b_meta["is_content_parallel"]].reset_index(drop=True)
            ja_meta = parallel_meta[parallel_meta["side"] == "JA"]
            ko_meta = parallel_meta[parallel_meta["side"] == "KO"]
            ja_idx = ja_meta["prompt_idx"].to_numpy()
            ko_idx = ko_meta["prompt_idx"].to_numpy()

            # B condition: paired (i, i). D condition: rotated (i, j=(i+4)%8).
            # Pre-reg locks N=8 and k=4 rotation; assert hard so a future filter
            # change cannot silently shift the D-condition definition.
            n_pairs = len(ja_idx)
            assert n_pairs == 8, (
                f"E1B primary lens expects exactly 8 content-parallel pairs; got {n_pairs}. "
                f"If you intend to run on a different N, update the rotation k explicitly."
            )
            j_idx = [(i + 4) % 8 for i in range(n_pairs)]

            # Encode B and D side activations at the trained layer.
            ja_at_L = e1b_h[ja_idx, layer, :]  # (8, 2048)
            ko_at_L = e1b_h[ko_idx, layer, :]
            ko_rot_at_L = ko_at_L[j_idx]

            # Compute pair-wise feature activations:
            # B[i] = (z(ja[i]) + z(ko[i])) / 2 — joint activation for same-meaning pair
            # D[i] = (z(ja[i]) + z(ko_rot[i])) / 2 — joint for diff-meaning rotated
            z_ja = encode_with_sae(ckpt, ja_at_L)
            z_ko = encode_with_sae(ckpt, ko_at_L)
            z_ko_rot = encode_with_sae(ckpt, ko_rot_at_L)
            B_act = (z_ja + z_ko) / 2  # (8, dict_size)
            D_act = (z_ja + z_ko_rot) / 2

            # Discrimination score = mean_i (B_act - D_act).
            mean_B = B_act.mean(axis=0)  # (dict_size,)
            mean_D = D_act.mean(axis=0)
            discrim = mean_B - mean_D
            for f in range(len(discrim)):
                discrim_rows.append({
                    "model_slug": slug, "layer": layer, "feature_idx": f,
                    "mean_B": float(mean_B[f]), "mean_D": float(mean_D[f]),
                    "discrim_score": float(discrim[f]),
                })

    df = pd.DataFrame(discrim_rows)
    out_path = args.output_root / args.run_tag / "experiment_e2_sae" / "feature_discrimination.csv"
    df.to_csv(out_path, index=False)
    LOGGER.info("Wrote %s (%d feature scores)", out_path, len(df))

    # Report top-k per (model, layer) and Base-vs-Global comparison.
    top_k = args.top_k or 16
    for (slug, layer), g in df.groupby(["model_slug", "layer"]):
        top = g.nlargest(top_k, "discrim_score")
        print(f"\n=== {slug} L{layer}: top-{top_k} discriminative features ===")
        print(top[["feature_idx", "mean_B", "mean_D", "discrim_score"]].to_string(index=False))

    # Installation-signature analysis: Global-SAE-L26 features that rank high
    # in Global but whose closest-by-decoder-cosine Base feature ranks low.
    #
    # Two SAEs trained on different activation distributions (Base vs Global)
    # produce non-aligned feature dictionaries. Joining by raw feature_idx is
    # semantically invalid because feature_idx 17 in Base may represent a
    # completely different direction than feature_idx 17 in Global. The
    # correct cross-SAE comparison matches each Global feature to its most
    # similar Base feature by decoder-column cosine similarity, then asks
    # whether that Base "equivalent" feature has a meaningful B-D score.
    L26 = df[df["layer"] == 26]
    if not L26.empty and len(L26["model_slug"].unique()) == 2:
        base_ckpt = load_sae_checkpoint(
            args.output_root / args.run_tag / "experiment_e2_sae"
            / "tiny-aya-base_L26" / "sae_checkpoint.pt"
        )
        glob_ckpt = load_sae_checkpoint(
            args.output_root / args.run_tag / "experiment_e2_sae"
            / "tiny-aya-global_L26" / "sae_checkpoint.pt"
        )
        # decoder_weight: (hidden, dict_size). Normalize columns to unit norm.
        dec_b = base_ckpt["decoder_weight"].numpy()
        dec_g = glob_ckpt["decoder_weight"].numpy()
        dec_b_n = dec_b / (np.linalg.norm(dec_b, axis=0, keepdims=True) + 1e-12)
        dec_g_n = dec_g / (np.linalg.norm(dec_g, axis=0, keepdims=True) + 1e-12)
        # Cosine similarity matrix: (dict_size_global, dict_size_base)
        cos_matrix = dec_g_n.T @ dec_b_n
        best_base = cos_matrix.argmax(axis=1)
        best_cos = cos_matrix.max(axis=1)

        base_scores = (
            L26[L26["model_slug"] == "tiny-aya-base"]
            .set_index("feature_idx")["discrim_score"].to_dict()
        )
        glob_scores = (
            L26[L26["model_slug"] == "tiny-aya-global"]
            .set_index("feature_idx")["discrim_score"].to_dict()
        )
        install_rows = []
        for f_g in range(dec_g.shape[1]):
            f_b = int(best_base[f_g])
            install_rows.append({
                "global_feature_idx": f_g,
                "matched_base_feature_idx": f_b,
                "decoder_cosine_similarity": float(best_cos[f_g]),
                "global_score": float(glob_scores.get(f_g, 0.0)),
                "base_matched_score": float(base_scores.get(f_b, 0.0)),
                "installation_score": (
                    float(glob_scores.get(f_g, 0.0)) - float(base_scores.get(f_b, 0.0))
                ),
            })
        install_df = pd.DataFrame(install_rows).sort_values(
            "installation_score", ascending=False,
        )
        install_path = (
            args.output_root / args.run_tag / "experiment_e2_sae"
            / "installation_signature_L26.csv"
        )
        install_df.to_csv(install_path, index=False)
        print(f"\n=== Top-{top_k} INSTALLATION-signature features at L=26 (cosine-matched) ===")
        print(install_df.head(top_k).to_string(index=False))
        # Diagnostic: distribution of decoder cosines (high = good alignment).
        print(
            f"\nDecoder-cosine match diagnostics (Global -> Base nearest):\n"
            f"  median cosine={np.median(best_cos):.3f}, "
            f"mean={best_cos.mean():.3f}, "
            f"frac_above_0.5={(best_cos > 0.5).mean():.3f}, "
            f"frac_above_0.8={(best_cos > 0.8).mean():.3f}"
        )


def run_max_activating(args: argparse.Namespace) -> None:
    """Find max-activating prompts for top features in each SAE."""
    e2_root = args.output_root / args.run_tag / "experiment_e2_sae"
    discrim_csv = e2_root / "feature_discrimination.csv"
    if not discrim_csv.exists():
        raise SystemExit(f"Run --discriminate first; missing {discrim_csv}")
    discrim = pd.read_csv(discrim_csv)

    out_dir = e2_root / "maximally_activating"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the activation corpus (text source for prompts).
    corpus_path = Path("data/sae_activation_corpus_v1/rows.jsonl")
    corpus_rows = [json.loads(line) for line in corpus_path.read_text().splitlines() if line.strip()]

    top_k = args.top_k or 16
    n_show = args.n_show or 8

    for (slug, layer), g in discrim.groupby(["model_slug", "layer"]):
        top_features = g.nlargest(top_k, "discrim_score")["feature_idx"].tolist()
        sae_dir = e2_root / f"{slug}_L{layer}"
        ckpt = load_sae_checkpoint(sae_dir / "sae_checkpoint.pt")
        # Re-encode each saved activation shard and find rows with max
        # activation on the top features.
        act_dir = (
            args.output_root / args.run_tag / "experiment_e2_activations"
            / slug / f"L{layer}"
        )
        if not act_dir.exists():
            LOGGER.warning("Activation dir missing: %s", act_dir)
            continue
        max_records: Dict[int, List[Any]] = {f: [] for f in top_features}

        for shard in sorted(act_dir.glob("activations_shard_*.npy")):
            arr = np.load(shard)
            meta_path = shard.parent / shard.name.replace("activations_", "metadata_").replace(".npy", ".parquet")
            meta = pd.read_parquet(meta_path)
            z = encode_with_sae(ckpt, arr)
            for f in top_features:
                f_acts = z[:, f]
                if f_acts.max() > 0:
                    top_local = f_acts.argsort()[-n_show:][::-1]
                    for li in top_local:
                        m = meta.iloc[li]
                        max_records[f].append({
                            "feature_idx": f,
                            "activation": float(f_acts[li]),
                            "row_id": m["row_id"], "source": m["source"],
                            "language": m["language"], "task_family": m["task_family"],
                            "token_idx": int(m["token_idx"]),
                        })

        # Sort each feature's records by activation desc and take top n_show.
        out_rows = []
        for f, records in max_records.items():
            records.sort(key=lambda r: -r["activation"])
            top_n = records[:n_show]
            out_rows.extend(top_n)
        out_path = out_dir / f"{slug}_L{layer}_top{top_k}_max_activating.csv"
        pd.DataFrame(out_rows).to_csv(out_path, index=False)
        LOGGER.info("Wrote %s", out_path)

        # Also write a markdown dump with prompt text for human labeling.
        md_path = out_dir / f"{slug}_L{layer}_top{top_k}_max_activating.md"
        lines = [f"# Max-activating prompts: {slug} L{layer} top-{top_k}\n"]
        for f in top_features:
            lines.append(f"\n## Feature {f}\n")
            for r in [x for x in out_rows if x["feature_idx"] == f]:
                row = next((cr for cr in corpus_rows if cr["row_id"] == r["row_id"]), None)
                if row is None:
                    continue
                preview = row["text"][:200].replace("\n", " ")
                lines.append(
                    f"- act={r['activation']:.3f} src={r['source']} lang={r['language']} "
                    f"family={r.get('task_family')} tok_idx={r['token_idx']}\n"
                    f"  text: {preview!r}\n"
                )
        md_path.write_text("\n".join(lines))
        LOGGER.info("Wrote %s", md_path)


def char_ngram_f_score(prediction: str, reference: str, max_n: int = 6, beta: float = 2.0) -> float:
    """Small chrF-style score without external metric dependencies."""
    pred = re.sub(r"\s+", " ", prediction.strip())
    ref = re.sub(r"\s+", " ", reference.strip())
    if not pred or not ref:
        return 0.0

    precisions = []
    recalls = []
    for n in range(1, max_n + 1):
        pred_counts = Counter(pred[i:i + n] for i in range(max(len(pred) - n + 1, 0)))
        ref_counts = Counter(ref[i:i + n] for i in range(max(len(ref) - n + 1, 0)))
        if not pred_counts or not ref_counts:
            continue
        overlap = sum((pred_counts & ref_counts).values())
        precisions.append(overlap / sum(pred_counts.values()))
        recalls.append(overlap / sum(ref_counts.values()))

    if not precisions or not recalls:
        return 0.0
    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta2 = beta * beta
    return (1 + beta2) * precision * recall / ((beta2 * precision) + recall)


def target_lang_script_ratio(text: str, side: str) -> float:
    """Return target-script fraction over alphabetic chars for JA or KO outputs."""
    chars = [c for c in text if c.isalpha()]
    if not chars:
        return 0.0
    if side == "JA":
        target = [
            c for c in chars
            if "\u3040" <= c <= "\u30ff" or "\u4e00" <= c <= "\u9fff"
        ]
    elif side == "KO":
        target = [c for c in chars if "\uac00" <= c <= "\ud7af"]
    else:
        target = []
    return len(target) / len(chars)


def continuation_metrics(
    text: str,
    side: str,
    global_continuation: Optional[str],
) -> Dict[str, float]:
    stripped = text.strip()
    try:
        json.loads(stripped)
        is_json_valid = 1.0
    except json.JSONDecodeError:
        is_json_valid = 0.0

    metrics = {
        "is_json_valid": is_json_valid,
        "has_quotation_wrap": float(stripped.startswith('"') and stripped.endswith('"')),
        "no_comma_compliance": float("," not in stripped),
        "target_lang_script_ratio": target_lang_script_ratio(stripped, side),
        "chrf_to_global": math.nan,
    }
    if global_continuation is not None:
        metrics["chrf_to_global"] = char_ngram_f_score(stripped, global_continuation)
    return metrics


def relevant_metrics_for_subtask(subtask: str) -> List[str]:
    if subtask == "json_format":
        return ["is_json_valid", "target_lang_script_ratio", "chrf_to_global"]
    if subtask == "quotation":
        return ["has_quotation_wrap", "target_lang_script_ratio", "chrf_to_global"]
    if subtask == "no_comma":
        return ["no_comma_compliance", "target_lang_script_ratio", "chrf_to_global"]
    return ["target_lang_script_ratio", "chrf_to_global"]


def load_global_continuations(e2_root: Path) -> Dict[Tuple[str, str, str], str]:
    """Load optional Global baseline continuations if a prior run exported them."""
    candidates = [
        e2_root / "global_continuations.csv",
        e2_root / "global_baseline_continuations.csv",
        e2_root / "steering_global_baselines.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        text_col = next(
            (
                col for col in [
                    "continuation",
                    "response",
                    "generated_text",
                    "global_continuation",
                    "baseline_continuation",
                ]
                if col in df.columns
            ),
            None,
        )
        if text_col is None:
            LOGGER.warning("Skipping %s: no continuation-like text column", path)
            continue
        out: Dict[Tuple[str, str, str], str] = {}
        for _, row in df.iterrows():
            side = str(row.get("side", "")).upper()
            subtask = str(row.get("subtask", ""))
            source_row_id = str(row.get("source_row_id", ""))
            if not side and "packet_row_id" in row:
                packet_row_id = str(row["packet_row_id"])
                if packet_row_id.startswith("ja-"):
                    side = "JA"
                elif packet_row_id.startswith("ko-"):
                    side = "KO"
            if side in {"JA", "KO"} and subtask and source_row_id:
                out[(subtask, source_row_id, side)] = str(row[text_col])
        LOGGER.info("Loaded %d Global baseline continuations from %s", len(out), path)
        return out
    LOGGER.warning("No Global continuation file found under %s; chrf_to_global will be NaN", e2_root)
    return {}


def select_top_features(discrim: pd.DataFrame, model_slug: str, layer: int, top_k: int) -> List[int]:
    df = discrim.copy()
    if "model_slug" in df.columns:
        df = df[df["model_slug"] == model_slug]
    if "layer" in df.columns:
        df = df[df["layer"] == layer]
    if df.empty:
        raise SystemExit(
            f"No discriminative features for model_slug={model_slug!r}, layer={layer}. "
            "Run --discriminate first or check feature_discrimination.csv."
        )
    score_col = "discrim_score" if "discrim_score" in df.columns else df.columns[-1]
    return [int(f) for f in df.nlargest(top_k, score_col)["feature_idx"].tolist()]


def resolve_device(requested_device: str) -> str:
    import torch

    if requested_device == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but unavailable; falling back to CPU")
        return "cpu"
    return requested_device


def greedy_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: Any,
    max_new_tokens: int = 160,
) -> str:
    import torch

    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id if hasattr(tokenizer, "eos_token_id") else 0,
        )
    new_tokens = output_ids[0, input_ids.shape[1]:].detach().cpu().tolist()
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def generate_with_feature_steering(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: Any,
    layer: int,
    feature_direction: Any,
    scale: float,
) -> str:
    import torch

    layer_idx = layer - 1
    if layer_idx < 0 or layer_idx >= len(model.model.layers):
        raise SystemExit(f"Layer {layer} is outside model.model.layers range")
    direction = feature_direction.to(device=device, dtype=next(model.parameters()).dtype)

    def steering_hook(_module: Any, _inputs: Any, output: Any) -> Any:
        delta = scale * direction.view(1, 1, -1)
        if torch.is_tensor(output):
            return output + delta
        if isinstance(output, tuple):
            return (output[0] + delta, *output[1:])
        raise TypeError(f"Unsupported layer output type for steering hook: {type(output)}")

    handle = model.model.layers[layer_idx].register_forward_hook(steering_hook)
    try:
        return greedy_generate(model, tokenizer, prompt, device=device, max_new_tokens=160)
    finally:
        handle.remove()


def run_steering(args: argparse.Namespace) -> None:
    """Steer top SAE feature directions and score Marco-MIF continuations."""
    import torch
    from scripts.analyze_triangulation import (  # type: ignore[import-not-found]
        format_chat_prompt,
        load_e1b_pairs,
    )
    from src.logit_lens import load_model_and_tokenizer  # type: ignore[import-not-found]

    if not args.model_id:
        raise SystemExit("--steering needs --model-id")
    e2_root = args.output_root / args.run_tag / "experiment_e2_sae"
    discrim_csv = e2_root / "feature_discrimination.csv"
    if not discrim_csv.exists():
        raise SystemExit(f"Run --discriminate first; missing {discrim_csv}")
    sae_dir = e2_root / f"{args.model_slug}_L{args.layer}"
    ckpt_path = sae_dir / "sae_checkpoint.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"SAE checkpoint missing: {ckpt_path}")

    discrim = pd.read_csv(discrim_csv)
    top_features = select_top_features(discrim, args.model_slug, args.layer, args.top_k)
    ckpt = load_sae_checkpoint(ckpt_path)
    decoder_weight = ckpt["decoder_weight"]
    if decoder_weight.shape[0] != ckpt["hidden"]:
        raise SystemExit(
            f"Unexpected decoder_weight shape {tuple(decoder_weight.shape)} "
            f"for hidden={ckpt['hidden']}"
        )

    pairs = [p for p in load_e1b_pairs(args.packet_path) if p.is_content_parallel]
    if len(pairs) != 8:
        raise SystemExit(f"Expected 8 content-parallel E1B pairs, got {len(pairs)}")

    global_continuations = load_global_continuations(e2_root)

    device_name = resolve_device(args.device)
    LOGGER.info("Loading model %s on %s ...", args.model_id, device_name)
    model, tokenizer = load_model_and_tokenizer(args.model_id, device=device_name)
    device = next(model.parameters()).device
    LOGGER.info("Model loaded on %s", device)

    baseline_cache: Dict[Tuple[int, str], str] = {}
    formatted_prompts: Dict[Tuple[int, str], str] = {}
    for pair in pairs:
        for side, prompt in (("JA", pair.ja_prompt), ("KO", pair.ko_prompt)):
            key = (pair.pair_idx, side)
            formatted = format_chat_prompt(tokenizer, prompt)
            formatted_prompts[key] = formatted
            baseline_cache[key] = greedy_generate(model, tokenizer, formatted, device=device, max_new_tokens=160)

    rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for feature_idx in top_features:
        direction = decoder_weight[:, feature_idx].detach().to(dtype=torch.float32)
        lifted_cases = 0
        total_cases = 0
        LOGGER.info("Steering feature_idx=%d scale=%s layer=%d", feature_idx, args.scale, args.layer)
        for pair in pairs:
            for side in ("JA", "KO"):
                key = (pair.pair_idx, side)
                baseline = baseline_cache[key]
                steered = generate_with_feature_steering(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=formatted_prompts[key],
                    device=device,
                    layer=args.layer,
                    feature_direction=direction,
                    scale=args.scale,
                )
                global_text = global_continuations.get((pair.subtask, pair.source_row_id, side))
                baseline_metrics = continuation_metrics(baseline, side, global_text)
                steered_metrics = continuation_metrics(steered, side, global_text)
                relevant = set(relevant_metrics_for_subtask(pair.subtask))
                case_lifted = False
                for metric_name in STEERING_METRICS:
                    baseline_value = baseline_metrics[metric_name]
                    steered_value = steered_metrics[metric_name]
                    lifted = bool(
                        metric_name in relevant
                        and not math.isnan(baseline_value)
                        and not math.isnan(steered_value)
                        and steered_value > baseline_value
                    )
                    case_lifted = case_lifted or lifted
                    rows.append({
                        "feature_idx": feature_idx,
                        "scale": args.scale,
                        "pair_idx": pair.pair_idx,
                        "side": side,
                        "subtask": pair.subtask,
                        "source_row_id": pair.source_row_id,
                        "metric_name": metric_name,
                        "baseline_value": baseline_value,
                        "steered_value": steered_value,
                        "lifted": lifted,
                    })
                total_cases += 1
                lifted_cases += int(case_lifted)
        summary_rows.append({
            "feature_idx": feature_idx,
            "scale": args.scale,
            "layer": args.layer,
            "lifted_cases": lifted_cases,
            "total_cases": total_cases,
        })

    out_path = e2_root / "steering_results.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    LOGGER.info("Wrote %s (%d metric rows)", out_path, len(rows))

    summary_path = e2_root / "steering_feature_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    LOGGER.info("Wrote %s", summary_path)
    print("\n=== Steering lift summary ===")
    print(pd.DataFrame(summary_rows).to_string(index=False))


def run_smoke(args: argparse.Namespace) -> None:
    """Synthetic-data smoke: verify discrimination computation works."""
    rng = np.random.default_rng(args.seed)
    # Fake activations: 2 models × 2 layers × 8 prompts × hidden=64
    # Synthetic SAE checkpoint
    import torch
    hidden = 64
    dict_size = 256
    k = 8
    enc_w = torch.tensor(rng.standard_normal((dict_size, hidden)).astype(np.float32))
    enc_b = torch.zeros(dict_size, dtype=torch.float32)
    dec_w = torch.tensor(rng.standard_normal((hidden, dict_size)).astype(np.float32))
    ckpt = {
        "encoder_weight": enc_w, "encoder_bias": enc_b,
        "decoder_weight": dec_w, "hidden": hidden, "dict_size": dict_size, "k": k,
    }
    x = rng.standard_normal((8, hidden)).astype(np.float32)
    z = encode_with_sae(ckpt, x)
    print(f"Smoke encode_with_sae: x shape={x.shape}, z shape={z.shape}, "
          f"avg active features per sample={(z>0).sum(axis=-1).mean():.1f}")
    print(f"  non-zero count == k? {(z>0).sum(axis=-1).max() == k}")
    LOGGER.info("Smoke OK.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--e1-run-tag", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--packet-path", type=Path, required=True,
                        help="JSONL path; row schema documented in README")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--n-show", type=int, default=8)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--layer", type=int, default=26)
    parser.add_argument("--model-slug", default="tiny-aya-base")
    parser.add_argument("--model-id", required=True,
                        help="HF id or local path (e.g., CohereLabs/tiny-aya-base)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    parser.add_argument("--discriminate", action="store_true")
    parser.add_argument("--max-activating", action="store_true")
    parser.add_argument("--steering", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.smoke:
        run_smoke(args)
        return 0
    if not args.run_tag:
        raise SystemExit("--run-tag is required for non-smoke modes")
    if args.discriminate:
        run_discriminate(args)
        return 0
    if args.max_activating:
        run_max_activating(args)
        return 0
    if args.steering:
        if args.top_k is None:
            args.top_k = 3
        run_steering(args)
        return 0
    raise SystemExit("Specify one of: --discriminate, --max-activating, --steering, --smoke")


if __name__ == "__main__":
    sys.exit(main())
