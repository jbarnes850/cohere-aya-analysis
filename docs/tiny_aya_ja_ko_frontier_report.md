# Tiny Aya JA/KO Frontier Post-Training Report

## Objective
Improve Japanese and Korean quality and efficiency on Tiny Aya Global by targeting decoding-stage behavior with late-layer LoRA SFT and conditional DPO.

## Scope
- Hardware: single H100
- Core budget: 8 hours for data build, baseline quick eval, SFT, optional DPO, post quick eval
- Extended budget: expanded evaluation and comparator inference allowed to exceed 8 hours

## Model and Data
- Base model: `models/tiny-aya-global` (or equivalent Hub ID)
- SFT data sources:
  - `CohereLabs/aya_dataset`
  - `CohereLabs/aya_collection`
  - `openlanguagedata/flores_plus`
  - `izumi-lab/llm-japanese-dataset`
- Eval sources:
  - `CohereLabs/Global-MMLU-Lite` (quick)
  - `CohereLabs/Global-MMLU` (expanded)
  - `openlanguagedata/flores_plus`
  - `CohereLabs/aya_evaluation_suite`
- Comparator models:
  - `google/gemma-3-4b-it`
  - `Qwen/Qwen3-4B-Instruct` (fallbacks allowed per config)

## Training Configuration
- Framework: TRL + PEFT + Transformers + Accelerate
- Precision: BF16
- LoRA target layers: 28-35
- Modules: `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`
- `modules_to_save`: `lm_head`
- CPT:
  - `max_seq_length=4096`
  - `per_device_train_batch_size=2`
  - `gradient_accumulation_steps=32`
  - `max_steps=4000`
  - `lr=1e-5`, cosine, warmup 2%
- SFT:
  - `max_seq_length=4096`
  - `per_device_train_batch_size=2`
  - `gradient_accumulation_steps=16`
  - `max_steps=1000`
  - `lr=1e-4`, cosine, warmup 5%
- DPO (conditional):
  - `max_steps=250`
  - `lr=2e-5`
  - `beta=0.1` (sigmoid loss)
- QLoRA fallback enabled on OOM

## Runtime Summary
- Run ID: `<fill>`
- Start UTC: `<fill>`
- End UTC: `<fill>`
- Core elapsed (min): `<fill>`
- DPO executed: `<yes/no>`
- Final adapter path: `<fill>`

## Results
### Quick Pre/Post
- Metrics file: `outputs/posttrain/<run_id>/metrics/quick_pre/metrics.csv`
- Metrics file: `outputs/posttrain/<run_id>/metrics/quick_post/metrics.csv`

### Expanded Pre/Post + Comparators
- Pre: `outputs/posttrain/<run_id>/metrics/expanded_pre/metrics.csv`
- Post: `outputs/posttrain/<run_id>/metrics/expanded_post/metrics.csv`
- Comparators: `outputs/posttrain/<run_id>/metrics/expanded_comparators/`
- Comparison summary: `outputs/posttrain/<run_id>/metrics/expanded_summary/decision_summary.json`

## Go / No-Go
- Target checks: `outputs/posttrain/<run_id>/metrics/expanded_summary/go_no_go_checks.csv`
- Decision: `<GO | CONDITIONAL_GO | NO_GO>`

## Interpretation
- JA quality delta: `<fill>`
- KO quality delta: `<fill>`
- FLORES chrF++ delta: `<fill>`
- Language confusion delta: `<fill>`
- Entity preservation delta: `<fill>`
- Structured validity delta: `<fill>`
- EN guardrail status: `<fill>`
- Frontier gap closure: `<fill>`

## Business Implications
1. `<fill>`
2. `<fill>`
3. `<fill>`

## Limitations
1. `<fill>`
2. `<fill>`
3. `<fill>`

## Reproducibility Appendix
- Git commit: `<fill>`
- Configs:
  - `training/configs/tiny_aya_ja_ko_cpt.yaml`
  - `training/configs/tiny_aya_ja_ko_sft.yaml`
  - `training/configs/tiny_aya_ja_ko_dpo.yaml`
  - `eval/configs/quick_8h.yaml`
  - `eval/configs/expanded_frontier.yaml`
- Pipeline state: `outputs/posttrain/<run_id>/artifacts/pipeline_state.json`
- Seeds: `42`
