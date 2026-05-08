#!/usr/bin/env python3
"""Experiment E2 step 2: train a TopK Sparse Autoencoder on saved activations.

Architecture (Gao et al. 2024 TopK SAE):
  - Encoder: Linear(hidden=2048 -> dict_size=16384), bias.
  - TopK activation: keep top k=32 features by magnitude per sample, zero rest.
  - Decoder: Linear(dict_size -> hidden), no bias on decoder by default.
  - Reconstruction loss: MSE between input activation and decoder(TopK(encoder(x))).
  - Auxiliary "AuxK" loss on dead features: keep features that haven't fired in
    last `aux_window` steps active by penalizing the top-k_aux dead features
    on a residual reconstruction objective.

Inputs: per-layer sharded activations from
  <output-root>/<run-tag>/experiment_e2_activations/<slug>/L<layer>/
    activations_shard_NNNN.npy

Outputs:
  <output-root>/<run-tag>/experiment_e2_sae/<slug>_L<layer>/
    sae_checkpoint.pt           encoder + decoder weights
    training_metrics.csv        per-step reconstruction MSE, dead-feature ratio
    sae_config.json             hyperparameters

Modes:
  --smoke            tiny CPU run (10 steps, 100 activations, dict=64) to verify loop
  default (GPU)      full training on saved activations

Usage:
  # GPU smoke (1 epoch, 1 model x 1 layer):
  python scripts/run_experiment_e2_train_sae.py \
    --model-slug tiny-aya-global --layer 26 \
    --run-tag e2_<timestamp> --device cuda --epochs 1
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np


LOGGER = logging.getLogger("e2_train_sae")

DEFAULT_OUTPUT_ROOT = Path("outputs/runs")
DEFAULT_K = 32
DEFAULT_DICT_SIZE = 16384
DEFAULT_HIDDEN = 2048
DEFAULT_LR = 1e-4
DEFAULT_BATCH = 4096
DEFAULT_EPOCHS = 5
AUX_K = 64
AUX_WINDOW = 200


def load_shards(layer_dir: Path) -> np.ndarray:
    """Concatenate all activation shards under layer_dir."""
    shards = sorted(layer_dir.glob("activations_shard_*.npy"))
    if not shards:
        raise SystemExit(f"No shards found under {layer_dir}")
    arrs = [np.load(s) for s in shards]
    out = np.concatenate(arrs, axis=0)
    LOGGER.info("Loaded %d shards from %s, total %d activations × %d hidden",
                len(shards), layer_dir, out.shape[0], out.shape[1])
    return out


def make_sae_module(hidden: int, dict_size: int, k: int):
    """Build a TopK SAE module. Lazy-import torch."""
    import torch
    import torch.nn as nn

    class TopKSAE(nn.Module):
        def __init__(self, hidden: int, dict_size: int, k: int):
            super().__init__()
            self.hidden = hidden
            self.dict_size = dict_size
            self.k = k
            self.encoder = nn.Linear(hidden, dict_size, bias=True)
            self.decoder = nn.Linear(dict_size, hidden, bias=False)
            # Initialize decoder rows to unit norm (standard practice).
            with torch.no_grad():
                w = self.decoder.weight  # (hidden, dict_size)
                w.copy_(w / (w.norm(dim=0, keepdim=True) + 1e-8))

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            # ReLU before TopK is an architectural variant from Gao et al. 2024
            # (which uses pre-ReLU TopK). Documented in the E2 report.
            return torch.relu(self.encoder(x))

        def topk(self, z: torch.Tensor, k: int) -> torch.Tensor:
            """Keep top-k features by magnitude per sample, zero the rest."""
            topk_vals, topk_idx = torch.topk(z, k, dim=-1)
            mask = torch.zeros_like(z)
            mask.scatter_(-1, topk_idx, topk_vals)
            return mask

        def forward(self, x: torch.Tensor):
            z = self.encode(x)
            z_topk = self.topk(z, self.k)
            x_hat = self.decoder(z_topk)
            return x_hat, z_topk, z

    return TopKSAE(hidden=hidden, dict_size=dict_size, k=k)


def train_sae(
    activations: np.ndarray, hidden: int, dict_size: int, k: int,
    epochs: int, batch_size: int, lr: float, device: str,
    out_dir: Path, seed: int = 0,
):
    import torch
    import torch.optim as optim

    # Deterministic seeding so Base-SAE and Global-SAE training runs share
    # the same initialization and data-permutation order. The Base-vs-Global
    # comparison still cannot rely on raw feature_idx alignment because the
    # two SAEs receive different activation distributions and converge to
    # different feature directions; downstream analysis must match features
    # by decoder-column cosine similarity, not by index. Seeding here only
    # rules out spurious init-induced differences.
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    sae = make_sae_module(hidden, dict_size, k).to(device)
    optimizer = optim.Adam(sae.parameters(), lr=lr)
    n_acts = activations.shape[0]
    LOGGER.info(
        "Training SAE: %d activations, hidden=%d, dict=%d, k=%d, epochs=%d, batch=%d, lr=%g, seed=%d",
        n_acts, hidden, dict_size, k, epochs, batch_size, lr, seed,
    )

    rng = np.random.default_rng(seed)
    last_fire_step = torch.zeros(dict_size, dtype=torch.long, device=device)
    metrics: List[dict] = []
    step = 0
    for epoch in range(epochs):
        perm = rng.permutation(n_acts)
        n_batches = max(1, n_acts // batch_size)
        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            x = torch.tensor(activations[idx], device=device, dtype=torch.float32)
            x_hat, z_topk, z_pre = sae(x)
            recon_loss = ((x_hat - x) ** 2).mean()

            # AuxK loss: pick top features that have NOT fired recently and
            # train them to reconstruct the residual.
            with torch.no_grad():
                fired = (z_topk > 0).any(dim=0)
                last_fire_step[fired] = step
                dead_mask = (step - last_fire_step) > AUX_WINDOW
                num_dead = int(dead_mask.sum().item())
            if num_dead > 0:
                residual = x - x_hat.detach()
                z_dead = z_pre.clone()
                z_dead[:, ~dead_mask] = 0.0
                topk_dead = min(AUX_K, num_dead)
                if topk_dead > 0:
                    z_dead = sae.topk(z_dead, topk_dead)
                    aux_recon = sae.decoder(z_dead)
                    aux_loss = ((aux_recon - residual) ** 2).mean()
                    loss = recon_loss + 0.03 * aux_loss
                else:
                    loss = recon_loss
            else:
                loss = recon_loss

            optimizer.zero_grad()
            loss.backward()
            # Standard practice: project decoder columns to unit norm.
            optimizer.step()
            with torch.no_grad():
                w = sae.decoder.weight
                w.div_(w.norm(dim=0, keepdim=True) + 1e-8)

            log_every = 50 if n_acts >= 50_000 else 5
            if step % log_every == 0:
                dead_ratio = float(((step - last_fire_step) > AUX_WINDOW).float().mean().item())
                metrics.append({
                    "step": step, "epoch": epoch, "batch": b,
                    "recon_mse": float(recon_loss.item()),
                    "dead_ratio": dead_ratio,
                    "num_dead": num_dead,
                })
                LOGGER.info(
                    "  step %d epoch %d batch %d: MSE=%.4f, dead_ratio=%.3f",
                    step, epoch, b, recon_loss.item(), dead_ratio,
                )
            step += 1

    # Save checkpoint and metrics.
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "encoder_weight": sae.encoder.weight.detach().cpu(),
        "encoder_bias": sae.encoder.bias.detach().cpu(),
        "decoder_weight": sae.decoder.weight.detach().cpu(),
        "hidden": hidden, "dict_size": dict_size, "k": k,
    }, out_dir / "sae_checkpoint.pt")

    import pandas as pd
    pd.DataFrame(metrics).to_csv(out_dir / "training_metrics.csv", index=False)
    (out_dir / "sae_config.json").write_text(json.dumps({
        "hidden": hidden, "dict_size": dict_size, "k": k, "epochs": epochs,
        "batch_size": batch_size, "lr": lr, "aux_k": AUX_K, "aux_window": AUX_WINDOW,
        "n_activations": int(n_acts),
    }, indent=2))
    LOGGER.info("Saved SAE checkpoint and metrics to %s", out_dir)


def run_smoke(args: argparse.Namespace) -> None:
    """Tiny CPU smoke: verify training loop works on synthetic data."""
    rng = np.random.default_rng(args.seed)
    n_acts = 1024
    hidden = 64
    dict_size = 256
    k = 8
    activations = rng.standard_normal((n_acts, hidden)).astype(np.float32)
    out_dir = Path("/tmp") / "e2_sae_smoke"
    train_sae(
        activations, hidden=hidden, dict_size=dict_size, k=k,
        epochs=args.epochs, batch_size=128, lr=DEFAULT_LR,
        device="cpu", out_dir=out_dir, seed=args.seed,
    )
    # Verify reconstruction MSE decreased.
    import pandas as pd
    df = pd.read_csv(out_dir / "training_metrics.csv")
    LOGGER.info(
        "Smoke verification: MSE first=%.4f, last=%.4f (%.1f%% decrease)",
        df["recon_mse"].iloc[0], df["recon_mse"].iloc[-1],
        100 * (1 - df["recon_mse"].iloc[-1] / df["recon_mse"].iloc[0]),
    )


def run_full(args: argparse.Namespace) -> None:
    if not args.run_tag or not args.model_slug or args.layer is None:
        raise SystemExit("Full training needs --run-tag, --model-slug, --layer")
    layer_dir = (
        args.output_root / args.run_tag / "experiment_e2_activations"
        / args.model_slug / f"L{args.layer}"
    )
    activations = load_shards(layer_dir)
    out_dir = (
        args.output_root / args.run_tag / "experiment_e2_sae"
        / f"{args.model_slug}_L{args.layer}"
    )
    train_sae(
        activations, hidden=DEFAULT_HIDDEN, dict_size=DEFAULT_DICT_SIZE,
        k=DEFAULT_K, epochs=args.epochs, batch_size=DEFAULT_BATCH, lr=DEFAULT_LR,
        device=args.device, out_dir=out_dir, seed=args.seed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-slug", default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.smoke:
        if args.epochs == DEFAULT_EPOCHS:
            args.epochs = 2  # small for smoke
        run_smoke(args)
        return 0
    run_full(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
