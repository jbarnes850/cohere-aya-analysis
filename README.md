# Multilingual Interpretability on Tiny Aya

Code accompanying an investigation of how Cohere's Tiny Aya model family carries multilingual meaning from input through internal representation to decoded behavior. The Tiny Aya base / global / water variants share an identical pre-training run and differ only in their post-training stage, which gives a controlled experimental handle for isolating post-training effects on the multilingual route.

These scripts produce the activations, paired representational measurements, sparse-autoencoder dictionaries, and residual-stream patches used in the working writeup at [jbarnes850.github.io](https://jbarnes850.github.io). The original eval rows are not bundled with this release; the schema below specifies what to provide so the pipeline runs on any Tiny-Aya-shaped corpus.

## Commands

| Script | Purpose |
|---|---|
| `build_triangulation_corpus.py` | Build the cross-language triangulation corpus from FLORES+ devtest (112 rows: 4 language pairs, 14 source indices, 2 sides). |
| `build_sae_activation_corpus.py` | Combine a user-provided eval-row packet with FLORES+ rows to produce the SAE activation corpus. |
| `analyze_triangulation.py` | Paired representational geometry across the three Tiny Aya checkpoints over the triangulation corpus. |
| `extract_activations.py` | Per-prompt residual-stream extraction at chosen layers; writes sharded NPY plus parquet metadata. |
| `train_sae.py` | TopK sparse-autoencoder training on saved activations, with an auxiliary dead-feature loss. |
| `analyze_sae_features.py` | Discrimination, max-activating prompts, and feature steering on trained SAEs. |
| `analyze_english_pivot.py` | Test the English-pivot hypothesis at the multilingual decoding stage. |
| `causal_validation.py` | Bidirectional residual-stream patching to test triangulation findings causally. |

## Library modules

| Module | Purpose |
|---|---|
| `src/causal_patching.py` | Patching primitives (token divergence, teacher-forced log-prob, patched log-prob). |
| `src/logit_lens.py` | Model loading and layer-by-layer logit-lens utilities for Aya-family causal LMs. |
| `src/prompts.py` | Multilingual prompt suite across en, ja, ko, zh, vi, id, th. |

## Quickstart

```bash
# Python 3.11+
pip install -r requirements.txt

# Build the FLORES+ derived corpus (CPU)
python scripts/build_triangulation_corpus.py

# Build the SAE activation corpus (requires your own eval-row JSONL; schema below)
python scripts/build_sae_activation_corpus.py --packet path/to/your_rows.jsonl

# Extract activations from a Tiny Aya checkpoint (GPU)
python scripts/extract_activations.py \
    --model-id CohereLabs/tiny-aya-base \
    --model-slug tiny-aya-base \
    --run-tag my_run \
    --device cuda
```

Every script accepts `--smoke` for a CPU sanity pass and `--help` for full options.

## Inputs

### Models

`CohereLabs/tiny-aya-base`, `CohereLabs/tiny-aya-global`, `CohereLabs/tiny-aya-water` on Hugging Face. Each script takes `--model-id` (or `--base-model-id` / `--global-model-id` for `causal_validation.py`) as a Hugging Face identifier or a local path to a materialized checkpoint.

### Datasets

`openlanguagedata/flores_plus`, loaded by the build scripts via `datasets.load_dataset`.

### Eval-row JSONL packet

Several scripts require an eval-row packet via `--packet` or `--packet-path`. Each row is a JSON object:

| Field | Type | Required | Description |
|---|---|---|---|
| `packet_row_id` | string | yes | Unique row id with a prefix from the table below. |
| `source_row_id` | string | yes | Upstream source-row identifier. |
| `prompt` | string | yes | Model input. |
| `language` | string | yes | Language code (`ja`, `ko`, `jpn_Jpan`, ...). |
| `task_family` | string | optional | Inferred from `packet_row_id` prefix when omitted. |

Recognized `packet_row_id` prefixes:

| Prefix | Subset |
|---|---|
| `ja-flores-`, `ko-flores-` | translation rows |
| `ja-mif-<subtask>-...`, `ko-mif-<subtask>-...` | paired Marco-MIF instruction-following rows |
| `ja-datapilot-...`, `ko-law-...`, `ko-legal-qa-...` | non-translation control rows |

`analyze_triangulation.py:load_e1b_pairs` enforces the original eval shape (17 paired Marco-MIF rows, 8 of which are content-parallel). Forking is recommended for other shapes.

## Runtime

- Python 3.11+.
- CUDA-capable GPU for extraction and patching. The Tiny Aya checkpoints fit on a single 24 GB GPU.
- CPU-only smoke modes are supported for end-to-end pipeline checks.

## Citation

Tiny Aya: Salamanca et al., *The Tiny Aya Series* (2026), [arXiv:2603.11510](https://arxiv.org/abs/2603.11510).

This analysis: working writeup at [jbarnes850.github.io](https://jbarnes850.github.io).

## License

MIT. See `LICENSE`.
