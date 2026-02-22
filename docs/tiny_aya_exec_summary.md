# Tiny Aya JA/KO Frontier Executive Summary

## Objective
Deliver a targeted post-training update for Tiny Aya to improve Japanese and Korean performance and efficiency, with comparator-backed expanded evaluation against Gemma 3 4B and Qwen3 4B.

## Approach
- Decoding-focused adaptation via late-layer LoRA SFT (layers 28-35 + `lm_head`)
- Conditional DPO phase for output-fidelity alignment
- Mandatory expanded evaluation on Global-MMLU JA/KO and FLORES+ JA/KO directions
- Frontier comparison vs Gemma 3 4B and Qwen3 4B

## Outcome Snapshot
- Decision: `<GO | CONDITIONAL_GO | NO_GO>`
- JA quality delta: `<fill>`
- KO quality delta: `<fill>`
- FLORES+ chrF++ delta: `<fill>`
- Language confusion relative change: `<fill>`
- Frontier gap closure: `<fill>`

## Recommendation
`<fill with next action based on decision>`

## Pointers
- Full technical report: `docs/tiny_aya_ja_ko_frontier_report.md`
- Final decision artifact: `outputs/posttrain/<run_id>/metrics/expanded_summary/decision_summary.json`
