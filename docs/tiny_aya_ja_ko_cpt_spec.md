# Tiny Aya JA/KO Frontier: Continued Pretraining Spec (v1)

Date: 2026-02-21  
Owner: Jarrod/Codex  
Scope: Tiny Aya 3B JA/KO frontier recovery with reproducible, gated training.

## 1) Direct Answer

Yes, we can do continued pretraining with TRL.

- Practical path: use `trl.SFTTrainer` in plain causal-LM mode on text sequences (`packing=true`, loss over all tokens).
- Caveat: TRL is viable for this scale; for very large throughput-optimized CPT, a custom `transformers`/`accelerate` loop can be more efficient.

## 2) Why CPT Is Needed in This Project

Current run pattern shows:

- Strong translation/format gains.
- JA/KO knowledge-style MCQ accuracy regressions.

That indicates post-training is over-optimizing output behavior without restoring enough parametric knowledge. This is a canonical CPT use case: restore/expand language knowledge first, then re-align behavior with SFT/DPO.

## 3) Primary Objective and Constraints

Objective:

- Recover and exceed JA/KO knowledge benchmarks while preserving translation gains and EN guardrails.

Hard constraints:

- No benchmark leakage.
- Deterministic eval (`temperature=0`, matched formatting).
- Staged validation before heavy GPU spend.
- Stop-loss gates on EN/control regressions.

## 4) Data Spec (CPT Mix)

Create a new CPT corpus artifact:

- `outputs/posttrain/ja_ko_cpt_v1/artifacts/data/train_text.jsonl`
- `outputs/posttrain/ja_ko_cpt_v1/artifacts/data/dev_text.jsonl`
- `outputs/posttrain/ja_ko_cpt_v1/artifacts/data/cpt_meta.yaml`

Target token mix (initial):

- JA knowledge text: 25%
- KO knowledge text: 25%
- JA/KO instructional + MCQ-style reasoning text: 15%
- EN replay (high-quality): 25%
- Helper multilingual transfer text (CJK-adjacent + high-quality multilingual): 10%

Rules:

- Dedup at document and near-duplicate levels.
- Language-ID + malformed filtering.
- Quality scoring/filtering before sampling.
- Explicit leakage blocklist: exclude any eval test content/IDs for Global-MMLU, FLORES devtest, and Aya eval suites.

## 5) Training Pipeline (Recommended)

### Phase A: CPT (new)

Method:

- TRL `SFTTrainer` over plain text (causal LM objective).
- Late-layer LoRA + `lm_head` first (safer anti-forgetting), then optional full-model CPT ablation.

Initial hyperparameters:

- `max_seq_length=4096`
- `packing=true`
- `per_device_train_batch_size=2`
- `gradient_accumulation_steps=32`
- `learning_rate=1e-5`
- `scheduler=cosine`
- `warmup_ratio=0.02`
- `max_steps=4000` (adjust after smoke throughput measurement)
- `bf16=true`
- `seed=42`

Checkpointing/evals:

- Save every 200 steps.
- Run quick dev gates every 400 steps.

### Phase B: SFT (existing, adjusted)

- Run current JA/KO SFT recipe on top of CPT adapter/base.
- Keep MCQ objective in selection, but use dev-only data (no test-split usage).

### Phase C: DPO (existing, reduced)

- Keep DPO short and error-focused (`100-150` steps).
- Preference pairs should emphasize hard negatives (wrong but plausible MCQ option) over easy malformed negatives.

## 6) Leakage and Evaluation-Parity Fixes (Mandatory)

Before rerun:

- Replace any `Global-MMLU-Lite split=test` usage in training-time selection/pref construction with dev-only or held-out train-dev split.
- Preserve `test` strictly for final quick/expanded reporting.
- Log eval parity explicitly per run:
  - prompt format
  - n-shot
  - decoding mode
  - max tokens
  - special token handling

## 7) Experiment Matrix

R0: Baseline lock  
- Re-run quick pre only for provenance if needed; else lock baseline artifact hashes.

R1: CPT-only smoke  
- `max_steps=200`, tiny subset.  
- Goal: stable loss, no immediate EN collapse.

R2: CPT-only full  
- Full CPT schedule.  
- Gate: JA/KO quick MMLU trend positive vs base; EN not beyond stop-loss.

R3: CPT + SFT  
- Existing SFT with leakage-safe MCQ selection.

R4: CPT + SFT + DPO  
- Reduced DPO schedule.  
- Final quick post; then expanded post if quick gates pass.

## 8) Stop-Loss and Go/No-Go Gates

Stop-loss (interrupt run):

- EN quick MMLU drops >1.0 absolute without JA or KO gain >=0.5 for two consecutive gates.
- Repeated language confusion regressions >25% relative.
- Training instability (loss divergence or NaN).

Quick gate to proceed to expanded post:

- JA delta >= +0.5 vs baseline.
- KO delta >= +0.5 vs baseline.
- EN regression <= 0.5 absolute.
- Structured validity and confusion non-regressive.

Final decision remains your locked project criteria (core + guardrails + frontier closure).

## 9) Implementation Notes for This Repo

Additions:

- `training/configs/tiny_aya_ja_ko_cpt.yaml`
- `training/train_cpt.py`

`train_cpt.py` should reuse patterns from:

- `training/train_sft.py`
- `training/train_dpo.py`

and switch dataset handling to plain text JSONL (`{"text": ...}`).

Validation order before GPU:

1. `ruff check`
2. `python3 -m py_compile training/train_cpt.py`
3. zero-pod dry run (`max_steps=2`, CPU sanity)
4. single-GPU smoke (`max_steps=20`)
5. full run

## 10) Why This Spec Is Research-Aligned

- ÜberWeb supports quality-first multilingual curation and bidirectional transfer effects.
- ATLAS supports empirical transfer-aware multilingual scaling decisions.
- TRL docs support SFT-based supervised LM continuation.

References:

- DatologyAI ÜberWeb blog: https://www.datologyai.com/blog/berweb-insights-from-multilingual-curation-for-a-20-trillion-token-dataset
- ÜberWeb paper: https://arxiv.org/abs/2602.15210
- ATLAS paper: https://arxiv.org/abs/2510.22037
- ATLAS blog: https://research.google/blog/atlas-practical-scaling-laws-for-multilingual-models/
- TRL SFT docs: https://huggingface.co/docs/trl/sft_trainer
