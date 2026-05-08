#!/usr/bin/env python3
"""Experiment E3: causal validation of E1 route findings.

Pre-registration: docs/experiment_e1_e2_prereg.md, E3 section.

Layer convention follows the C-series runner exactly:
output_hidden_states=True returns n_layers + 1 hidden states. A patch at
layer L hooks model.model.layers[L] and replaces the last-position residual
with donor.hidden_states[L + 1][0, -1, :].
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Make src.* and scripts.* imports resolve from repo root.
_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from src.causal_patching import (  # noqa: E402
    first_divergence,
    teacher_forced_continuation_logprob,
    teacher_forced_continuation_logprob_with_patch,
)
from scripts.analyze_triangulation import (  # noqa: E402
    E1B_CONTENT_PARALLEL,
    format_chat_prompt,
    load_e1b_pairs,
    load_e1c_rows,
)
from scripts.analyze_sae_features import (  # noqa: E402
    STEERING_METRICS,
    char_ngram_f_score,
    continuation_metrics,
    relevant_metrics_for_subtask,
    target_lang_script_ratio,
)


LOGGER = logging.getLogger("experiment_e3_causal_validation")

DEFAULT_OUTPUT_ROOT = Path("outputs/runs")

MARCO_LAYERS = (26, 20, 34)
KO_LAW_LAYERS = (21, 26, 34)
ISOLATION_LAYERS = (21, 26, 34)
JOINT_LAYERS = (20, 22, 24, 26)
MAX_NEW_TOKENS = 160
CONTINUATION_TOKENS = 8


@dataclass(frozen=True)
class PatchCase:
    case_id: str
    task: str
    prompt: str
    side: str
    subtask: str
    source_row_id: str
    reference_text: str | None = None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def resolve_device(requested_device: str) -> str:
    import torch

    if requested_device == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but unavailable; falling back to CPU")
        return "cpu"
    return requested_device


def load_packet_rows(packet_path: Path) -> dict[str, dict[str, Any]]:
    return {row["packet_row_id"]: row for row in read_jsonl(packet_path)}


def marco_mif_cases(packet_path: Path) -> list[PatchCase]:
    pairs = [pair for pair in load_e1b_pairs(packet_path) if pair.is_content_parallel]
    if len(pairs) != 8:
        raise SystemExit(f"Expected 8 content-parallel Marco-MIF pairs, got {len(pairs)}")

    cases: list[PatchCase] = []
    for pair in pairs:
        if (pair.subtask, pair.source_row_id) not in E1B_CONTENT_PARALLEL:
            raise SystemExit(f"Unexpected non-parallel Marco-MIF pair: {pair}")
        for side, prompt in (("JA", pair.ja_prompt), ("KO", pair.ko_prompt)):
            cases.append(
                PatchCase(
                    case_id=f"mif-{pair.subtask}-{pair.source_row_id}-{side.lower()}",
                    task="marco_mif",
                    prompt=prompt,
                    side=side,
                    subtask=pair.subtask,
                    source_row_id=pair.source_row_id,
                )
            )
    if len(cases) != 16:
        raise SystemExit(f"Expected 16 Marco-MIF side rows, got {len(cases)}")
    return cases


def ko_law_cases(packet_path: Path) -> list[PatchCase]:
    rows = [
        row
        for row in load_e1c_rows(packet_path)
        if row.packet_row_id.startswith("ko-law-")
    ]
    if len(rows) != 12:
        raise SystemExit(f"Expected 12 ko-law rows, got {len(rows)}")
    return [
        PatchCase(
            case_id=row.packet_row_id,
            task="ko_law",
            prompt=row.prompt,
            side="KO",
            subtask="ko_law",
            source_row_id=row.source_row_id,
        )
        for row in rows
    ]


def translation_cases(packet_path: Path) -> list[PatchCase]:
    packet = load_packet_rows(packet_path)
    cases: list[PatchCase] = []
    for source_row_id in range(14):
        for side, prefix in (("JA", "ja-flores"), ("KO", "ko-flores")):
            row_id = f"{prefix}-{source_row_id}"
            row = packet.get(row_id)
            if row is None:
                raise SystemExit(f"Missing translation row in v2 packet: {row_id}")
            cases.append(
                PatchCase(
                    case_id=row_id,
                    task="translation",
                    prompt=row["prompt"],
                    side=side,
                    subtask="translation_calibration",
                    source_row_id=str(source_row_id),
                    reference_text=row.get("reference_text"),
                )
            )
    if len(cases) != 28:
        raise SystemExit(f"Expected 28 JA/KO FLORES side rows, got {len(cases)}")
    return cases


def encode_ids(tokenizer: Any, text: str, device: Any) -> Any:
    return tokenizer.encode(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)


def decode_new_tokens(tokenizer: Any, output_ids: Any, prompt_len: int) -> str:
    new_tokens = output_ids[0, prompt_len:].detach().cpu().tolist()
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def capture_donor_residual(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: Any,
    layer: int,
) -> Any:
    import torch

    input_ids = encode_ids(tokenizer, prompt, device)
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True, use_cache=False)
    return out.hidden_states[layer + 1][0, -1, :].detach().clone()


def capture_many_donor_residuals(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: Any,
    layers: tuple[int, ...],
) -> dict[int, Any]:
    import torch

    input_ids = encode_ids(tokenizer, prompt, device)
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True, use_cache=False)
    return {
        layer: out.hidden_states[layer + 1][0, -1, :].detach().clone()
        for layer in layers
    }


def make_prompt_end_patch_hook(donor_last_pos: Any, prompt_len: int) -> Any:
    import torch

    patched_once = False

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        nonlocal patched_once
        hidden = output[0] if isinstance(output, tuple) else output
        if (not patched_once) and hidden.shape[1] == prompt_len:
            patched_once = True
            patched_hidden = hidden.clone()
            patched_hidden[0, prompt_len - 1, :] = donor_last_pos.to(
                dtype=patched_hidden.dtype,
                device=patched_hidden.device,
            )
            if torch.is_tensor(output):
                return patched_hidden
            return (patched_hidden, *output[1:])
        return output

    return hook


def make_full_sequence_position_hook(donor_position: Any, position: int) -> Any:
    import torch

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        patched_hidden = hidden.clone()
        patched_hidden[0, position, :] = donor_position.to(
            dtype=patched_hidden.dtype,
            device=patched_hidden.device,
        )
        if torch.is_tensor(output):
            return patched_hidden
        return (patched_hidden, *output[1:])

    return hook


def greedy_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: Any,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    import torch

    input_ids = encode_ids(tokenizer, prompt, device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            use_cache=True,
            pad_token_id=getattr(tokenizer, "eos_token_id", None) or 0,
        )
    return decode_new_tokens(tokenizer, output_ids, input_ids.shape[1])


def greedy_generate_with_prompt_patch(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: Any,
    layer: int,
    donor_last_pos: Any,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    import torch

    input_ids = encode_ids(tokenizer, prompt, device)
    hook = make_prompt_end_patch_hook(donor_last_pos, prompt_len=input_ids.shape[1])
    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                use_cache=True,
                pad_token_id=getattr(tokenizer, "eos_token_id", None) or 0,
            )
        return decode_new_tokens(tokenizer, output_ids, input_ids.shape[1])
    finally:
        handle.remove()


def greedy_generate_with_joint_prompt_patch(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: Any,
    donors_by_layer: dict[int, Any],
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    import torch

    input_ids = encode_ids(tokenizer, prompt, device)
    handles = []
    try:
        for layer, donor in donors_by_layer.items():
            hook = make_prompt_end_patch_hook(donor, prompt_len=input_ids.shape[1])
            handles.append(model.model.layers[layer].register_forward_hook(hook))
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                use_cache=True,
                pad_token_id=getattr(tokenizer, "eos_token_id", None) or 0,
            )
        return decode_new_tokens(tokenizer, output_ids, input_ids.shape[1])
    finally:
        for handle in handles:
            handle.remove()


def forward_last_logits(model: Any, input_ids: Any) -> tuple[Any, Any]:
    import torch

    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True, use_cache=False)
    return out.logits[0, -1, :].float(), out.hidden_states


def forward_last_logits_only(model: Any, input_ids: Any) -> Any:
    import torch

    with torch.no_grad():
        out = model(input_ids, use_cache=False)
    return out.logits[0, -1, :].float()


def forward_logits_with_position_patch(
    model: Any,
    input_ids: Any,
    layer: int,
    position: int,
    donor_position: Any,
) -> Any:
    import torch

    hook = make_full_sequence_position_hook(donor_position, position)
    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            out = model(input_ids, use_cache=False)
        return out.logits[0, -1, :].float()
    finally:
        handle.remove()


def teacher_forced_per_token_logprobs(
    model: Any,
    prompt_ids: Any,
    continuation_ids: list[int],
) -> list[float]:
    import torch
    import torch.nn.functional as F

    if not continuation_ids:
        return []
    cont = torch.tensor([continuation_ids], device=prompt_ids.device)
    full = torch.cat([prompt_ids, cont], dim=1)
    with torch.no_grad():
        out = model(full, use_cache=False)
    log_probs = F.log_softmax(out.logits[0].float(), dim=-1)
    n_prompt = prompt_ids.shape[1]
    return [
        float(log_probs[n_prompt - 1 + pos, tok].cpu())
        for pos, tok in enumerate(continuation_ids)
    ]


def teacher_forced_per_token_logprobs_with_patch(
    model: Any,
    prompt_ids: Any,
    continuation_ids: list[int],
    layer: int,
    donor_last_pos: Any,
) -> list[float]:
    import torch
    import torch.nn.functional as F

    if not continuation_ids:
        return []
    cont = torch.tensor([continuation_ids], device=prompt_ids.device)
    full = torch.cat([prompt_ids, cont], dim=1)
    prompt_end = prompt_ids.shape[1] - 1
    hook = make_full_sequence_position_hook(donor_last_pos, prompt_end)
    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            out = model(full, use_cache=False)
        log_probs = F.log_softmax(out.logits[0].float(), dim=-1)
        n_prompt = prompt_ids.shape[1]
        return [
            float(log_probs[n_prompt - 1 + pos, tok].cpu())
            for pos, tok in enumerate(continuation_ids)
        ]
    finally:
        handle.remove()


def first_n_token_ids(tokenizer: Any, text: str, n_tokens: int) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)[:n_tokens]


def mean_relevant_metric_delta(
    subtask: str,
    baseline_metrics: dict[str, float],
    patched_metrics: dict[str, float],
) -> float:
    deltas = []
    for name in relevant_metrics_for_subtask(subtask):
        before = baseline_metrics.get(name, math.nan)
        after = patched_metrics.get(name, math.nan)
        if math.isnan(before) or math.isnan(after):
            continue
        deltas.append(after - before)
    return float(np.mean(deltas)) if deltas else math.nan


def add_marco_metric_rows(
    rows: list[dict[str, Any]],
    case: PatchCase,
    direction: str,
    layer_label: str,
    layer: int | None,
    baseline_text: str,
    patched_text: str,
    global_text: str,
) -> None:
    baseline_metrics = continuation_metrics(baseline_text, case.side, global_text)
    patched_metrics = continuation_metrics(patched_text, case.side, global_text)
    relevant = set(relevant_metrics_for_subtask(case.subtask))
    for metric_name in STEERING_METRICS:
        before = baseline_metrics[metric_name]
        after = patched_metrics[metric_name]
        rows.append(
            {
                "case_id": case.case_id,
                "task": case.task,
                "direction": direction,
                "layer": layer,
                "layer_label": layer_label,
                "side": case.side,
                "subtask": case.subtask,
                "source_row_id": case.source_row_id,
                "metric_name": metric_name,
                "baseline_value": before,
                "patched_value": after,
                "effect": after - before
                if not (math.isnan(before) or math.isnan(after))
                else math.nan,
                "is_relevant_metric": metric_name in relevant,
                "baseline_text": baseline_text,
                "patched_text": patched_text,
                "global_baseline_text": global_text,
            }
        )


def branch_margin_rows(
    case: PatchCase,
    tokenizer: Any,
    device: Any,
    recipient_model: Any,
    donor_model: Any,
    recipient_baseline_text: str,
    base_text: str,
    global_text: str,
    layer: int,
    direction: str,
) -> dict[str, Any] | None:
    prompt_ids = tokenizer.encode(
        case.prompt,
        add_special_tokens=False,
    )
    base_ids = tokenizer.encode(base_text, add_special_tokens=False)
    global_ids = tokenizer.encode(global_text, add_special_tokens=False)
    div = first_divergence(base_ids, global_ids)
    if div is None:
        return None
    if div >= len(base_ids) or div >= len(global_ids):
        return None

    import torch

    div_input_ids = torch.tensor([prompt_ids + base_ids[:div]], device=device)
    recipient_logits = forward_last_logits_only(recipient_model, div_input_ids)
    _, donor_hs = forward_last_logits(donor_model, div_input_ids)
    donor = donor_hs[layer + 1][0, -1, :].detach().clone()
    base_token = base_ids[div]
    global_token = global_ids[div]
    baseline_margin = float(recipient_logits[global_token] - recipient_logits[base_token])
    patched_logits = forward_logits_with_position_patch(
        recipient_model,
        div_input_ids,
        layer,
        div_input_ids.shape[1] - 1,
        donor,
    )
    patched_margin = float(patched_logits[global_token] - patched_logits[base_token])
    return {
        "case_id": case.case_id,
        "task": case.task,
        "direction": direction,
        "layer": layer,
        "side": case.side,
        "subtask": case.subtask,
        "source_row_id": case.source_row_id,
        "metric_name": "branch_margin",
        "baseline_value": baseline_margin,
        "patched_value": patched_margin,
        "effect": patched_margin - baseline_margin,
        "first_divergence_offset": div,
        "base_token_at_div": base_token,
        "global_token_at_div": global_token,
        "recipient_baseline_text": recipient_baseline_text,
        "base_baseline_text": base_text,
        "global_baseline_text": global_text,
    }


def summarize_isolation(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    effect_col = "effect"
    if "metric_name" in df.columns:
        grouped = df.groupby(["layer", "task", "direction"], dropna=False)
    else:
        grouped = df.groupby(["layer", "task", "direction"], dropna=False)
    summary = grouped[effect_col].agg(["count", "mean", "median", "std"]).reset_index()
    summary = summary.rename(
        columns={
            "count": "n_observations",
            "mean": "mean_effect",
            "median": "median_effect",
            "std": "std_effect",
        }
    )
    return summary


def load_models_once(args: argparse.Namespace) -> tuple[Any, Any, Any, Any]:
    from src.logit_lens import load_model_and_tokenizer

    if not args.base_model_id or not args.global_model_id:
        raise SystemExit(
            "--base-model-id and --global-model-id are required outside --smoke"
        )
    device_name = resolve_device(args.device)
    LOGGER.info("Loading Base model %s on %s", args.base_model_id, device_name)
    base_model, tokenizer = load_model_and_tokenizer(args.base_model_id, device=device_name)
    LOGGER.info("Loading Global model %s on %s", args.global_model_id, device_name)
    global_model, _ = load_model_and_tokenizer(args.global_model_id, device=device_name)
    device = next(base_model.parameters()).device
    return base_model, global_model, tokenizer, device


def baseline_key(model_slug: str, case_id: str) -> tuple[str, str]:
    return (model_slug, case_id)


def build_baseline_cache(
    cases: list[PatchCase],
    base_model: Any,
    global_model: Any,
    tokenizer: Any,
    device: Any,
) -> dict[tuple[str, str], str]:
    cache: dict[tuple[str, str], str] = {}
    for i, case in enumerate(cases, 1):
        LOGGER.info("Baseline generation %d/%d: %s", i, len(cases), case.case_id)
        prompt = format_chat_prompt(tokenizer, case.prompt)
        cache[baseline_key("base", case.case_id)] = greedy_generate(
            base_model,
            tokenizer,
            prompt,
            device,
        )
        cache[baseline_key("global", case.case_id)] = greedy_generate(
            global_model,
            tokenizer,
            prompt,
            device,
        )
    return cache


def run_e3a(
    args: argparse.Namespace,
    base_model: Any,
    global_model: Any,
    tokenizer: Any,
    device: Any,
    baseline_cache: dict[tuple[str, str], str] | None = None,
) -> pd.DataFrame:
    cases = marco_mif_cases(args.packet_path)
    if baseline_cache is None:
        baseline_cache = build_baseline_cache(
            cases,
            base_model,
            global_model,
            tokenizer,
            device,
        )

    rows: list[dict[str, Any]] = []
    for case in cases:
        prompt = format_chat_prompt(tokenizer, case.prompt)
        base_text = baseline_cache[baseline_key("base", case.case_id)]
        global_text = baseline_cache[baseline_key("global", case.case_id)]
        for layer in MARCO_LAYERS:
            global_donor = capture_donor_residual(
                global_model,
                tokenizer,
                prompt,
                device,
                layer,
            )
            patched_base = greedy_generate_with_prompt_patch(
                base_model,
                tokenizer,
                prompt,
                device,
                layer,
                global_donor,
            )
            add_marco_metric_rows(
                rows,
                case,
                "global_to_base",
                f"L{layer}",
                layer,
                base_text,
                patched_base,
                global_text,
            )

            base_donor = capture_donor_residual(
                base_model,
                tokenizer,
                prompt,
                device,
                layer,
            )
            patched_global = greedy_generate_with_prompt_patch(
                global_model,
                tokenizer,
                prompt,
                device,
                layer,
                base_donor,
            )
            add_marco_metric_rows(
                rows,
                case,
                "base_to_global",
                f"L{layer}",
                layer,
                global_text,
                patched_global,
                global_text,
            )

    df = pd.DataFrame(rows)
    out_dir = args.output_root / args.run_tag / "experiment_e3"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "e3a_marco_mif_results.csv", index=False)
    LOGGER.info("Wrote %s", out_dir / "e3a_marco_mif_results.csv")
    return df


def run_e3b(
    args: argparse.Namespace,
    base_model: Any,
    global_model: Any,
    tokenizer: Any,
    device: Any,
    baseline_cache: dict[tuple[str, str], str] | None = None,
) -> pd.DataFrame:
    cases = ko_law_cases(args.packet_path)
    if baseline_cache is None:
        baseline_cache = build_baseline_cache(
            cases,
            base_model,
            global_model,
            tokenizer,
            device,
        )

    rows: list[dict[str, Any]] = []
    for case in cases:
        prompt = format_chat_prompt(tokenizer, case.prompt)
        prompt_ids = encode_ids(tokenizer, prompt, device)
        base_text = baseline_cache[baseline_key("base", case.case_id)]
        base_cont = first_n_token_ids(tokenizer, base_text, CONTINUATION_TOKENS)
        if not base_cont:
            LOGGER.warning("Skipping %s: empty Base greedy continuation", case.case_id)
            continue
        for layer in KO_LAW_LAYERS:
            _, base_hs = forward_last_logits(base_model, prompt_ids)
            _, global_hs = forward_last_logits(global_model, prompt_ids)
            base_donor = base_hs[layer + 1][0, -1, :].detach().clone()
            global_donor = global_hs[layer + 1][0, -1, :].detach().clone()

            base_baseline = teacher_forced_continuation_logprob(
                base_model,
                prompt_ids,
                base_cont,
            )
            base_patched = teacher_forced_continuation_logprob_with_patch(
                base_model,
                prompt_ids,
                base_cont,
                layer,
                global_donor,
            )
            rows.append(
                {
                    "case_id": case.case_id,
                    "task": case.task,
                    "direction": "global_to_base",
                    "layer": layer,
                    "metric_name": "continuation_logprob_shift",
                    "baseline_value": base_baseline,
                    "patched_value": base_patched,
                    "continuation_logprob_shift": base_patched - base_baseline,
                    "effect": base_patched - base_baseline,
                    "base_greedy_text": base_text,
                    "continuation_n_tokens": len(base_cont),
                }
            )

            global_baseline = teacher_forced_continuation_logprob(
                global_model,
                prompt_ids,
                base_cont,
            )
            global_patched = teacher_forced_continuation_logprob_with_patch(
                global_model,
                prompt_ids,
                base_cont,
                layer,
                base_donor,
            )
            rows.append(
                {
                    "case_id": case.case_id,
                    "task": case.task,
                    "direction": "base_to_global",
                    "layer": layer,
                    "metric_name": "continuation_logprob_shift",
                    "baseline_value": global_baseline,
                    "patched_value": global_patched,
                    "continuation_logprob_shift": global_patched - global_baseline,
                    "effect": global_patched - global_baseline,
                    "base_greedy_text": base_text,
                    "continuation_n_tokens": len(base_cont),
                }
            )

    df = pd.DataFrame(rows)
    out_dir = args.output_root / args.run_tag / "experiment_e3"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "e3b_ko_law_results.csv", index=False)
    LOGGER.info("Wrote %s", out_dir / "e3b_ko_law_results.csv")
    return df


def run_e3c(
    args: argparse.Namespace,
    base_model: Any,
    global_model: Any,
    tokenizer: Any,
    device: Any,
) -> pd.DataFrame:
    marco_cases = marco_mif_cases(args.packet_path)
    law_cases = ko_law_cases(args.packet_path)
    trans_cases = translation_cases(args.packet_path)
    all_cases = marco_cases + law_cases + trans_cases
    baseline_cache = build_baseline_cache(
        all_cases,
        base_model,
        global_model,
        tokenizer,
        device,
    )

    matrix_rows: list[dict[str, Any]] = []
    for case in marco_cases:
        prompt = format_chat_prompt(tokenizer, case.prompt)
        base_text = baseline_cache[baseline_key("base", case.case_id)]
        global_text = baseline_cache[baseline_key("global", case.case_id)]
        for layer in ISOLATION_LAYERS:
            global_donor = capture_donor_residual(
                global_model,
                tokenizer,
                prompt,
                device,
                layer,
            )
            patched_base = greedy_generate_with_prompt_patch(
                base_model,
                tokenizer,
                prompt,
                device,
                layer,
                global_donor,
            )
            base_metrics = continuation_metrics(base_text, case.side, global_text)
            patched_metrics = continuation_metrics(patched_base, case.side, global_text)
            matrix_rows.append(
                {
                    "case_id": case.case_id,
                    "task": "marco_mif",
                    "direction": "global_to_base",
                    "layer": layer,
                    "metric_name": "mean_relevant_metric_delta",
                    "effect": mean_relevant_metric_delta(
                        case.subtask,
                        base_metrics,
                        patched_metrics,
                    ),
                }
            )

            base_donor = capture_donor_residual(
                base_model,
                tokenizer,
                prompt,
                device,
                layer,
            )
            patched_global = greedy_generate_with_prompt_patch(
                global_model,
                tokenizer,
                prompt,
                device,
                layer,
                base_donor,
            )
            global_metrics = continuation_metrics(global_text, case.side, global_text)
            patched_global_metrics = continuation_metrics(
                patched_global,
                case.side,
                global_text,
            )
            matrix_rows.append(
                {
                    "case_id": case.case_id,
                    "task": "marco_mif",
                    "direction": "base_to_global",
                    "layer": layer,
                    "metric_name": "mean_relevant_metric_delta",
                    "effect": mean_relevant_metric_delta(
                        case.subtask,
                        global_metrics,
                        patched_global_metrics,
                    ),
                }
            )

    for case in law_cases:
        prompt = format_chat_prompt(tokenizer, case.prompt)
        prompt_ids = encode_ids(tokenizer, prompt, device)
        base_text = baseline_cache[baseline_key("base", case.case_id)]
        cont = first_n_token_ids(tokenizer, base_text, CONTINUATION_TOKENS)
        if not cont:
            continue
        for layer in ISOLATION_LAYERS:
            _, base_hs = forward_last_logits(base_model, prompt_ids)
            _, global_hs = forward_last_logits(global_model, prompt_ids)
            base_donor = base_hs[layer + 1][0, -1, :].detach().clone()
            global_donor = global_hs[layer + 1][0, -1, :].detach().clone()

            base_lp = teacher_forced_continuation_logprob(base_model, prompt_ids, cont)
            patched_base_lp = teacher_forced_continuation_logprob_with_patch(
                base_model,
                prompt_ids,
                cont,
                layer,
                global_donor,
            )
            matrix_rows.append(
                {
                    "case_id": case.case_id,
                    "task": "ko_law",
                    "direction": "global_to_base",
                    "layer": layer,
                    "metric_name": "continuation_logprob_shift",
                    "effect": patched_base_lp - base_lp,
                }
            )

            global_lp = teacher_forced_continuation_logprob(global_model, prompt_ids, cont)
            patched_global_lp = teacher_forced_continuation_logprob_with_patch(
                global_model,
                prompt_ids,
                cont,
                layer,
                base_donor,
            )
            matrix_rows.append(
                {
                    "case_id": case.case_id,
                    "task": "ko_law",
                    "direction": "base_to_global",
                    "layer": layer,
                    "metric_name": "continuation_logprob_shift",
                    "effect": patched_global_lp - global_lp,
                }
            )

    for case in trans_cases:
        prompt = format_chat_prompt(tokenizer, case.prompt)
        base_text = baseline_cache[baseline_key("base", case.case_id)]
        global_text = baseline_cache[baseline_key("global", case.case_id)]
        for layer in ISOLATION_LAYERS:
            g2b = branch_margin_rows(
                case,
                tokenizer,
                device,
                base_model,
                global_model,
                base_text,
                base_text,
                global_text,
                layer,
                "global_to_base",
            )
            if g2b is not None:
                matrix_rows.append(g2b)
            b2g = branch_margin_rows(
                case,
                tokenizer,
                device,
                global_model,
                base_model,
                global_text,
                base_text,
                global_text,
                layer,
                "base_to_global",
            )
            if b2g is not None:
                matrix_rows.append(b2g)

    out_dir = args.output_root / args.run_tag / "experiment_e3"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix_rows).to_csv(out_dir / "e3c_isolation_matrix_raw.csv", index=False)
    summary = summarize_isolation(matrix_rows)
    summary.to_csv(out_dir / "e3c_isolation_matrix.csv", index=False)
    LOGGER.info("Wrote %s", out_dir / "e3c_isolation_matrix.csv")
    return summary


def run_per_token(
    args: argparse.Namespace,
    base_model: Any,
    global_model: Any,
    tokenizer: Any,
    device: Any,
) -> pd.DataFrame:
    cases = marco_mif_cases(args.packet_path)
    baseline_cache = build_baseline_cache(
        cases,
        base_model,
        global_model,
        tokenizer,
        device,
    )
    rows: list[dict[str, Any]] = []
    layer = 26
    for case in cases:
        prompt = format_chat_prompt(tokenizer, case.prompt)
        prompt_ids = encode_ids(tokenizer, prompt, device)
        global_text = baseline_cache[baseline_key("global", case.case_id)]
        target_ids = first_n_token_ids(tokenizer, global_text, CONTINUATION_TOKENS)
        if not target_ids:
            continue
        _, global_hs = forward_last_logits(global_model, prompt_ids)
        global_donor = global_hs[layer + 1][0, -1, :].detach().clone()
        base_lp = teacher_forced_per_token_logprobs(base_model, prompt_ids, target_ids)
        patched_lp = teacher_forced_per_token_logprobs_with_patch(
            base_model,
            prompt_ids,
            target_ids,
            layer,
            global_donor,
        )
        for pos, (before, after) in enumerate(zip(base_lp, patched_lp)):
            rows.append(
                {
                    "case_id": case.case_id,
                    "task": "marco_mif",
                    "direction": "global_to_base",
                    "layer": layer,
                    "position": pos,
                    "target_token_id": target_ids[pos],
                    "target_token_text": tokenizer.decode([target_ids[pos]]),
                    "baseline_logprob": before,
                    "patched_logprob": after,
                    "logprob_shift": after - before,
                }
            )
    df = pd.DataFrame(rows)
    out_dir = args.output_root / args.run_tag / "experiment_e3"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "e3_per_token_attribution.csv", index=False)
    LOGGER.info("Wrote %s", out_dir / "e3_per_token_attribution.csv")
    return df


def run_multi_layer(
    args: argparse.Namespace,
    base_model: Any,
    global_model: Any,
    tokenizer: Any,
    device: Any,
) -> pd.DataFrame:
    cases = marco_mif_cases(args.packet_path)
    baseline_cache = build_baseline_cache(
        cases,
        base_model,
        global_model,
        tokenizer,
        device,
    )
    rows: list[dict[str, Any]] = []
    for case in cases:
        prompt = format_chat_prompt(tokenizer, case.prompt)
        base_text = baseline_cache[baseline_key("base", case.case_id)]
        global_text = baseline_cache[baseline_key("global", case.case_id)]

        global_donor_l26 = capture_donor_residual(
            global_model,
            tokenizer,
            prompt,
            device,
            26,
        )
        single = greedy_generate_with_prompt_patch(
            base_model,
            tokenizer,
            prompt,
            device,
            26,
            global_donor_l26,
        )
        add_marco_metric_rows(
            rows,
            case,
            "global_to_base",
            "single_L26",
            26,
            base_text,
            single,
            global_text,
        )

        donors = capture_many_donor_residuals(
            global_model,
            tokenizer,
            prompt,
            device,
            JOINT_LAYERS,
        )
        joint = greedy_generate_with_joint_prompt_patch(
            base_model,
            tokenizer,
            prompt,
            device,
            donors,
        )
        add_marco_metric_rows(
            rows,
            case,
            "global_to_base",
            "joint_L20_22_24_26",
            None,
            base_text,
            joint,
            global_text,
        )

    df = pd.DataFrame(rows)
    out_dir = args.output_root / args.run_tag / "experiment_e3"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "e3_multi_layer_joint.csv", index=False)
    LOGGER.info("Wrote %s", out_dir / "e3_multi_layer_joint.csv")
    return df


def write_summary(args: argparse.Namespace) -> None:
    out_dir = args.output_root / args.run_tag / "experiment_e3"
    summary: dict[str, Any] = {
        "run_tag": args.run_tag,
        "acceptance_bands": {
            "e3a": "Global->Base moves Base toward Global on at least 5/8 paired cases; "
            "Base->Global degrades Global on at least 5/8 paired cases.",
            "e3b": "L21 ko-law positive/degrading effects on at least 5/12 cases; "
            "L26 ko-law null on <=2/12 cases.",
            "e3c": "Clean task-conditional route if at least 6/9 matrix cells "
            "match preregistered prediction.",
        },
        "files": sorted(path.name for path in out_dir.glob("*.csv")),
    }

    e3a_path = out_dir / "e3a_marco_mif_results.csv"
    if e3a_path.exists():
        e3a = pd.read_csv(e3a_path)
        rel = e3a[e3a["is_relevant_metric"]].copy()
        summary["e3a_mean_relevant_effect"] = (
            rel.groupby(["direction", "layer"])["effect"].mean().reset_index().to_dict("records")
        )

    e3b_path = out_dir / "e3b_ko_law_results.csv"
    if e3b_path.exists():
        e3b = pd.read_csv(e3b_path)
        summary["e3b_mean_logprob_shift"] = (
            e3b.groupby(["direction", "layer"])["continuation_logprob_shift"]
            .mean()
            .reset_index()
            .to_dict("records")
        )

    e3c_path = out_dir / "e3c_isolation_matrix.csv"
    if e3c_path.exists():
        summary["e3c_isolation_matrix"] = pd.read_csv(e3c_path).to_dict("records")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "e3_summary.json").write_text(json.dumps(summary, indent=2))
    LOGGER.info("Wrote %s", out_dir / "e3_summary.json")


def run_smoke() -> None:
    metrics = continuation_metrics('{"answer": "東京"}', "JA", '{"answer": "東京"}')
    assert metrics["is_json_valid"] == 1.0
    assert metrics["target_lang_script_ratio"] > 0.0
    assert metrics["chrf_to_global"] > 0.0

    quoted = continuation_metrics('"인터넷은 역사적으로 중요하다"', "KO", None)
    assert quoted["has_quotation_wrap"] == 1.0
    assert quoted["target_lang_script_ratio"] > 0.0

    no_comma = continuation_metrics("comma free Korean sentence", "KO", None)
    assert no_comma["no_comma_compliance"] == 1.0

    score = char_ngram_f_score("abcdef", "abcxyz")
    script_ratio = target_lang_script_ratio("한국어 English", "KO")
    relevant = relevant_metrics_for_subtask("json_format")
    print(
        "Smoke OK: "
        f"json_valid={metrics['is_json_valid']}, "
        f"quotation_wrap={quoted['has_quotation_wrap']}, "
        f"no_comma={no_comma['no_comma_compliance']}, "
        f"chrf={score:.4f}, "
        f"ko_script_ratio={script_ratio:.4f}, "
        f"json_relevant={','.join(relevant)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--e3a", action="store_true")
    parser.add_argument("--e3b", action="store_true")
    parser.add_argument("--e3c", action="store_true")
    parser.add_argument("--per-token", action="store_true")
    parser.add_argument("--multi-layer", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--base-model-id", default=None,
                        help="HF id or local path; required outside --smoke (e.g., CohereLabs/tiny-aya-base)")
    parser.add_argument("--global-model-id", default=None,
                        help="HF id or local path; required outside --smoke (e.g., CohereLabs/tiny-aya-global)")
    parser.add_argument("--packet-path", type=Path, default=None,
                        help="JSONL path; required outside --smoke (schema in README)")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.run_tag is None:
        args.run_tag = f"e3_{time.strftime('%Y%m%dT%H%M%S')}"

    if args.smoke:
        run_smoke()
        return 0

    selected = [
        args.e3a,
        args.e3b,
        args.e3c,
        args.per_token,
        args.multi_layer,
        args.all,
    ]
    if not any(selected):
        raise SystemExit("Specify --e3a, --e3b, --e3c, --per-token, --multi-layer, or --all")

    base_model, global_model, tokenizer, device = load_models_once(args)

    if args.all or args.e3a:
        run_e3a(args, base_model, global_model, tokenizer, device)
    if args.all or args.e3b:
        run_e3b(args, base_model, global_model, tokenizer, device)
    if args.all or args.e3c:
        run_e3c(args, base_model, global_model, tokenizer, device)
    if args.all or args.per_token:
        run_per_token(args, base_model, global_model, tokenizer, device)
    if args.all or args.multi_layer:
        run_multi_layer(args, base_model, global_model, tokenizer, device)
    write_summary(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
