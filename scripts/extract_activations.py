#!/usr/bin/env python3
"""Experiment E2 step 1: extract residual-stream activations at L=26 and L=30
from one Tiny Aya model on the SAE activation corpus.

Forwards each corpus prompt through the model with `output_hidden_states=True`
and saves last-position-of-each-token activations at the two target layers.
Tokens are accumulated across prompts and written as a single sharded
NPY file per (model, layer).

Output:
  <output>/<run-tag>/experiment_e2_activations/<slug>/L<layer>/
    activations_shard_NNNN.npy    (N_tokens_in_shard, hidden_size=2048) float32
    activations_metadata.parquet  per-token metadata: row_id, token_idx, source, ...

Usage:
  # GPU forward extraction:
  python scripts/run_experiment_e2_extract_activations.py \
    --model-id CohereLabs/tiny-aya-base \
    --model-slug tiny-aya-base \
    --device cuda \
    --run-tag e2_<timestamp>

  # CPU smoke (1 prompt, 1 layer):
  python scripts/run_experiment_e2_extract_activations.py --smoke
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Make src.* importable.
_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))


LOGGER = logging.getLogger("e2_extract_activations")

DEFAULT_CORPUS = Path("data/sae_activation_corpus_v1/rows.jsonl")
DEFAULT_OUTPUT_ROOT = Path("outputs/runs")

E2_LAYERS = [26, 30]
SHARD_TOKENS = 100_000  # write a new shard every 100K tokens
MAX_TOKENS_PER_PROMPT = 512  # truncate to keep memory bounded


def load_corpus(corpus_path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in corpus_path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def run_extract(args: argparse.Namespace) -> None:
    import torch
    from src.logit_lens import load_model_and_tokenizer  # type: ignore[import-not-found]

    if not args.model_id or not args.model_slug:
        raise SystemExit("--model-id and --model-slug required for extraction")

    out_root = (
        args.output_root / args.run_tag / "experiment_e2_activations" / args.model_slug
    )
    out_root.mkdir(parents=True, exist_ok=True)
    for L in E2_LAYERS:
        (out_root / f"L{L}").mkdir(parents=True, exist_ok=True)

    rows = load_corpus(args.corpus)
    LOGGER.info("Loaded %d corpus rows from %s", len(rows), args.corpus)

    LOGGER.info("Loading model %s on %s ...", args.model_id, args.device)
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(args.model_id, device=args.device)
    LOGGER.info("Model loaded in %.1fs", time.time() - t0)
    device = next(model.parameters()).device

    shard_buf: Dict[int, List[np.ndarray]] = {L: [] for L in E2_LAYERS}
    shard_meta: Dict[int, List[Dict[str, Any]]] = {L: [] for L in E2_LAYERS}
    shard_idx: Dict[int, int] = {L: 0 for L in E2_LAYERS}
    shard_tokens: Dict[int, int] = {L: 0 for L in E2_LAYERS}
    total_tokens = 0

    for row_i, r in enumerate(rows):
        text = r["text"]
        ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=False).to(device)
        # Truncate to keep memory bounded.
        if ids.shape[1] > MAX_TOKENS_PER_PROMPT:
            ids = ids[:, :MAX_TOKENS_PER_PROMPT]
        with torch.no_grad():
            out = model(ids, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  # tuple of length n_layers + 1
        n_tokens = ids.shape[1]
        for L in E2_LAYERS:
            # Hidden states at layer L (post-block), shape (1, n_tokens, hidden).
            h_L = hs[L][0].detach().to("cpu", dtype=torch.float32).numpy()  # (n_tokens, hidden)
            shard_buf[L].append(h_L)
            for t in range(n_tokens):
                shard_meta[L].append({
                    "row_idx": row_i,
                    "row_id": r["row_id"],
                    "source": r["source"],
                    "language": r["language"],
                    "task_family": r.get("task_family"),
                    "is_enterprise_prompt": r["is_enterprise_prompt"],
                    "token_idx": t,
                })
            shard_tokens[L] += n_tokens
            if shard_tokens[L] >= SHARD_TOKENS:
                _flush_shard(out_root, L, shard_idx[L], shard_buf[L], shard_meta[L])
                shard_idx[L] += 1
                shard_buf[L] = []
                shard_meta[L] = []
                shard_tokens[L] = 0
        total_tokens += n_tokens
        if (row_i + 1) % 200 == 0:
            LOGGER.info(
                "  %d/%d prompts, %d tokens accumulated",
                row_i + 1, len(rows), total_tokens,
            )

    # Flush remaining shards.
    for L in E2_LAYERS:
        if shard_buf[L]:
            _flush_shard(out_root, L, shard_idx[L], shard_buf[L], shard_meta[L])

    # Write summary index.
    summary = {
        "model_slug": args.model_slug,
        "n_prompts": len(rows),
        "total_tokens": total_tokens,
        "layers": E2_LAYERS,
        "shard_tokens_target": SHARD_TOKENS,
        "max_tokens_per_prompt": MAX_TOKENS_PER_PROMPT,
        "shards_per_layer": {str(L): shard_idx[L] + (1 if shard_tokens[L] > 0 else 0) for L in E2_LAYERS},
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    LOGGER.info("Extraction complete. %d tokens × %d layers extracted.", total_tokens, len(E2_LAYERS))


def _flush_shard(
    out_root: Path, layer: int, shard_idx: int,
    buf: List[np.ndarray], meta: List[Dict[str, Any]],
) -> None:
    import pandas as pd
    arr = np.concatenate(buf, axis=0).astype(np.float32)
    shard_path = out_root / f"L{layer}" / f"activations_shard_{shard_idx:04d}.npy"
    np.save(shard_path, arr)
    meta_path = out_root / f"L{layer}" / f"metadata_shard_{shard_idx:04d}.parquet"
    pd.DataFrame(meta).to_parquet(meta_path, index=False)
    LOGGER.info("Wrote shard %s shape=%s", shard_path, arr.shape)


def run_smoke(args: argparse.Namespace) -> None:
    """Synthetic smoke: verify the buffer→shard pipeline works without GPU."""
    out_root = Path("/tmp") / "e2_smoke" / "tiny-aya-base"
    out_root.mkdir(parents=True, exist_ok=True)
    for L in E2_LAYERS:
        (out_root / f"L{L}").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    fake_buf: Dict[int, List[np.ndarray]] = {L: [] for L in E2_LAYERS}
    fake_meta: Dict[int, List[Dict[str, Any]]] = {L: [] for L in E2_LAYERS}
    n_prompts = 5
    for row_i in range(n_prompts):
        n_tokens = rng.integers(20, 60)
        for L in E2_LAYERS:
            fake_buf[L].append(rng.standard_normal((n_tokens, 2048)).astype(np.float32))
            for t in range(n_tokens):
                fake_meta[L].append({
                    "row_idx": row_i, "row_id": f"smoke-{row_i}",
                    "source": "smoke", "language": "test",
                    "task_family": None, "is_enterprise_prompt": False,
                    "token_idx": t,
                })
    for L in E2_LAYERS:
        _flush_shard(out_root, L, 0, fake_buf[L], fake_meta[L])
    LOGGER.info("Smoke OK. Wrote shards under %s", out_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=None,
                        help="HF id or local path; required outside --smoke (e.g., CohereLabs/tiny-aya-base)")
    parser.add_argument("--model-slug", default=None)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    parser.add_argument("--smoke", action="store_true",
                        help="CPU synthetic smoke; no model needed")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.smoke:
        run_smoke(args)
        return 0
    if not args.run_tag:
        args.run_tag = "e2_" + time.strftime("%Y%m%d-%H%M%S")
    run_extract(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
