# Multilingual Representation Analysis: Aya Expanse 8B

Interpretability analysis examining how Aya Expanse 8B processes multilingual content internally, with Japanese (JA) and Korean (KO) as the primary case study.

**[Read the full technical report](docs/report.md)**

---

## Motivation

Aya Expanse 8B shows strong multilingual generation performance on benchmarks like m-ArenaHard and FLORES, but CJK (Chinese/Japanese/Korean) languages trail Arabic and other high-resource languages in head-to-head evaluations. This analysis uses interpretability methods to understand *where* in the network that gap originates — information that informs training strategy.

The core question: Is the CJK deficit a **representation problem** (the model lacks good internal understanding) or a **decoding problem** (the model understands but struggles to produce correct output)?

## Approach

We apply three complementary interpretability methods:

| Method | What it measures |
|--------|------------------|
| **CKA (Centered Kernel Alignment)** | Whether hidden states share the same geometric structure across languages |
| **Logit Lens** | Which layer the model "knows" the answer vs. commits to a surface form |
| **Entropy Curve** | Whether Aya follows the expected three-phase multilingual processing pattern |

These methods are applied to 70 semantically equivalent prompts across 7 languages (EN, JA, KO, ZH, VI, ID, TH) spanning 10 categories.

## Key Findings

1. **Representations are aligned.** CKA between JA-EN is 0.93-0.99 at every layer. The model builds equivalent internal representations regardless of input language.

2. **The gap is in late-layer decoding.** Non-English languages require 3-6 extra layers after concept emergence to produce the correct surface form. English averages 1.3 extra layers; JA averages 4.0.

3. **JA-KO show interference.** The JA-KO pair shows larger late-layer CKA divergence than any CJK-EN pair, suggesting Japanese and Korean may compete for decoding resources.

![CKA Representation Similarity](outputs/figures/representation_similarity.png)
*Cross-lingual CKA remains above 0.93 at all layers. The late-layer dip (layer 31) is where language-specific decoding occurs.*

The full analysis with methodology, tables, and figures is in **[docs/report.md](docs/report.md)**.

---

## Reproducing the Analysis

### Requirements

- Python 3.10+
- NVIDIA GPU with ~20GB VRAM (A100 recommended)
- 32GB+ system RAM
- ~20GB storage for model weights

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full analysis
python run_analysis.py --device cuda

# Run tokenizer analysis only (CPU, no GPU required)
python run_analysis.py --skip-logit-lens --skip-cka --device cpu
```

### Repository Structure

```
cohere-aya-analysis/
├── run_analysis.py        # Single entry point
├── requirements.txt       # Dependencies
├── src/
│   ├── prompts.py                    # Multilingual prompt suite
│   ├── tokenizer_analysis.py         # Token fertility, entity splitting
│   ├── logit_lens.py                 # Layer-by-layer analysis
│   ├── representation_similarity.py  # CKA computation
│   └── figures.py                    # Visualization
├── outputs/
│   ├── figures/           # Generated visualizations (gitignored)
│   └── tables/            # Raw data (CSV, gitignored)
└── docs/
    ├── report.md          # Frozen report (public artifact)
    └── figures/           # Frozen figures
```

### Output Files

| File | Description |
|------|-------------|
| `docs/report.md` | Frozen technical report (public artifact) |
| `docs/figures/*.png` | Frozen figures used in the report |
| `outputs/tables/*.csv` | Raw data for all analyses (generated) |
| `outputs/figures/*.png` | Visualizations (generated) |

### Frontier Post-Training Pipeline (Tiny Aya JA/KO)

This repository also includes a full TRL-based post-training and evaluation pipeline for Tiny Aya JA/KO optimization:

- `training/build_cpt_dataset.py` - builds leak-safe CPT text corpus from SFT pools + MCQ dev augmentation
- `training/train_cpt.py` - late-layer LoRA continued pretraining (CPT)
- `training/build_sft_dataset.py` - builds weighted JA/KO SFT mixture
- `training/train_sft.py` - late-layer LoRA SFT (with QLoRA fallback)
- `training/build_pref_dataset.py` - builds preference pairs for DPO
- `training/train_dpo.py` - conditional DPO stage
- `eval/run_eval_suite.py` - quick and expanded pre/post evaluation
- `eval/run_comparators.py` - expanded comparator inference
- `eval/compare_pre_post.py` - computes deltas, gap closure, and GO/NO-GO
- `training/run_frontier_pipeline.py` - end-to-end orchestration

Configs:

- `training/configs/tiny_aya_ja_ko_cpt.yaml`
- `training/configs/tiny_aya_ja_ko_sft.yaml`
- `training/configs/tiny_aya_ja_ko_dpo.yaml`
- `eval/configs/quick_8h.yaml`
- `eval/configs/expanded_frontier.yaml`

Current SFT data recipe highlights:

- Uses disjoint FLORES+ splits to prevent leakage (`dev` for training translation pool, `devtest` for eval).
- Adds high-supply open JA/KO instruction sources (`tellarin-ai/llm-japanese-dataset-vanilla-aya-format`, `heegyu/open-korean-instructions`, `beomi/KoAlpaca-v1.1a`).
- Adds open JA↔KO translation source (`sappho192/Tatoeba-Challenge-jpn-kor`) on top of FLORES-derived translation rows.

Run end-to-end:

```bash
python training/run_frontier_pipeline.py \
  --output-root outputs/posttrain
```

Expected outputs:

- `outputs/posttrain/<run_id>/metrics/quick_pre/`
- `outputs/posttrain/<run_id>/metrics/quick_post/`
- `outputs/posttrain/<run_id>/metrics/expanded_pre/`
- `outputs/posttrain/<run_id>/metrics/expanded_post/`
- `outputs/posttrain/<run_id>/metrics/expanded_comparators/`
- `outputs/posttrain/<run_id>/metrics/expanded_summary/`

---

## Model & Terms

Model weights and tokenizer: [CohereLabs/aya-expanse-8b](https://huggingface.co/CohereLabs/aya-expanse-8b). Use is subject to the terms and license in the model card.

## Data Sources

This analysis uses the following datasets. Please review and follow their terms:

- [openlanguagedata/flores_plus](https://huggingface.co/datasets/openlanguagedata/flores_plus) (FLORES+ devtest for non-JA fertility samples)
- [izumi-lab/llm-japanese-dataset](https://huggingface.co/datasets/izumi-lab/llm-japanese-dataset) (JA fertility samples)

## References

1. Dang et al. (2024). "Aya Expanse: Combining Research Breakthroughs for a New Multilingual Frontier." [arXiv:2412.04261](https://arxiv.org/abs/2412.04261)
2. Harrasse et al. (2025). "Tracing Multilingual Representations in LLMs with Cross-Layer Transcoders." [arXiv:2511.10840](https://arxiv.org/abs/2511.10840)
3. TranslateGemma Technical Report (Google 2026). [arXiv:2601.09012](https://arxiv.org/abs/2601.09012)
4. Kornblith et al. (2019). "Similarity of Neural Network Representations Revisited." [arXiv:1905.00414](https://arxiv.org/abs/1905.00414)

## License

See `LICENSE`.
