#!/usr/bin/env python3
"""Aya Expanse 8B analysis entry point."""

import argparse
import gc
import logging
import os
import sys
from pathlib import Path

import torch
sys.path.insert(0, str(Path(__file__).parent))

logger = logging.getLogger(__name__)


def run_analysis(args):
    """Run analysis pipeline."""
    results = {}

    logger.info("=" * 60)
    logger.info("Step 1: Tokenizer Analysis")
    logger.info("=" * 60)
    from src.tokenizer_analysis import run_tokenizer_analysis

    tokenizer_results = run_tokenizer_analysis(
        output_dir=args.output_dir,
        n_ja_samples=args.n_ja_samples,
        use_japanese_dataset=args.use_japanese_dataset,
    )
    results["tokenizer"] = tokenizer_results

    needs_gpu = not args.skip_logit_lens or not args.skip_cka
    model, tokenizer = None, None

    if needs_gpu:
        from src.logit_lens import load_model_and_tokenizer
        model, tokenizer = load_model_and_tokenizer(device=args.device)

    if not args.skip_logit_lens and model is not None:
        logger.info("=" * 60)
        logger.info("Step 2: Logit Lens Analysis")
        logger.info("=" * 60)
        from src.logit_lens import run_logit_lens_with_model

        logit_lens_results = run_logit_lens_with_model(
            model, tokenizer,
            output_dir=args.output_dir,
            device=args.device,
        )
        results["logit_lens"] = logit_lens_results

    if not args.skip_cka and model is not None:
        logger.info("=" * 60)
        logger.info("Step 3: CKA Representation Similarity")
        logger.info("=" * 60)
        from src.representation_similarity import run_representation_similarity

        cka_results = run_representation_similarity(
            output_dir=args.output_dir,
            device=args.device,
            model=model,
            tokenizer=tokenizer,
        )
        results["cka"] = cka_results

    if model is not None:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("=" * 60)
    logger.info("Step 4: Generating Figures")
    logger.info("=" * 60)
    from src.figures import generate_all_figures

    figure_paths = generate_all_figures(
        tokenizer_results=results.get("tokenizer"),
        logit_lens_results=results.get("logit_lens"),
        cka_results=results.get("cka"),
        overlap_results=None,
        switching_results=None,
        output_dir=args.output_dir,
    )

    print(f"\n{'=' * 60}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 60}")
    print(f"Figures: {len(figure_paths)} generated")
    print(f"Output directory: {args.output_dir}")
    print(f"Report: {args.output_dir}/report.md")


def main():
    parser = argparse.ArgumentParser(
        description="Aya Expanse 8B Multilingual Representation Analysis"
    )
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Analysis options
    parser.add_argument("--skip-logit-lens", action="store_true", help="Skip logit lens (GPU-heavy)")
    parser.add_argument("--skip-cka", action="store_true", help="Skip CKA similarity (GPU-heavy)")
    parser.add_argument("--n-ja-samples", type=int, default=1000, help="Japanese dataset samples")
    parser.add_argument("--use-japanese-dataset", action="store_true", default=True)
    parser.add_argument("--no-japanese-dataset", action="store_false", dest="use_japanese_dataset")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.output_dir, "analysis.log"), mode="w"),
        ],
    )

    logger.info("Starting Aya Expanse 8B Multilingual Analysis")
    logger.info("Device: %s", args.device)
    logger.info("Output: %s", args.output_dir)

    run_analysis(args)


if __name__ == "__main__":
    main()
