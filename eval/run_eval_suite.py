#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
from datasets import load_dataset
from sacrebleu.metrics import CHRF
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.frontier_utils import (
    MetricRow,
    detect_script_language,
    ensure_dir,
    language_confused,
    load_yaml,
    normalize_lang,
    now_utc_iso,
)


FLORES_LANG_KEYS = {
    "en": ["eng_Latn", "sentence_eng_Latn", "en", "english"],
    "ja": ["jpn_Jpan", "sentence_jpn_Jpan", "ja", "japanese"],
    "ko": ["kor_Hang", "sentence_kor_Hang", "ko", "korean"],
}


ENTITY_EVAL_SET = [
    {"lang": "ja", "entity": "富士通", "prompt": "次の企業名をそのまま含む1文を書いてください: 富士通"},
    {"lang": "ja", "entity": "トヨタ", "prompt": "次の企業名をそのまま含む1文を書いてください: トヨタ"},
    {"lang": "ja", "entity": "LGエレクトロニクス", "prompt": "次の企業名をそのまま含む1文を書いてください: LGエレクトロニクス"},
    {"lang": "ko", "entity": "삼성전자", "prompt": "다음 기업명을 그대로 포함한 한 문장을 작성하세요: 삼성전자"},
    {"lang": "ko", "entity": "토요타", "prompt": "다음 기업명을 그대로 포함한 한 문장을 작성하세요: 토요타"},
    {"lang": "ko", "entity": "LG전자", "prompt": "다음 기업명을 그대로 포함한 한 문장을 작성하세요: LG전자"},
]


STRUCTURED_EVAL_SET = [
    {
        "lang": "ja",
        "prompt": "JSONのみで回答してください。keys: company, revenue_2024_jpy, revenue_2025_jpy, yoy_growth_pct。company=富士通, revenue_2024_jpy=500000000, revenue_2025_jpy=560000000",
        "required_keys": ["company", "revenue_2024_jpy", "revenue_2025_jpy", "yoy_growth_pct"],
    },
    {
        "lang": "ko",
        "prompt": "JSON으로만 답하세요. keys: company, revenue_2024_jpy, revenue_2025_jpy, yoy_growth_pct. company=삼성전자, revenue_2024_jpy=700000000, revenue_2025_jpy=756000000",
        "required_keys": ["company", "revenue_2024_jpy", "revenue_2025_jpy", "yoy_growth_pct"],
    },
]


def _extract_first_present(row: Dict[str, Any], candidates: Sequence[str]) -> Optional[str]:
    for key in candidates:
        if key in row and row[key] is not None and str(row[key]).strip():
            return str(row[key])
        lower_key = key.lower()
        for row_key in row.keys():
            if row_key.lower() == lower_key and row[row_key] is not None and str(row[row_key]).strip():
                return str(row[row_key])
    return None


def _load_model_tokenizer(model_id: str, adapter_dir: Optional[str], attn_impl: str, bf16: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = dict(
        trust_remote_code=True,
        device_map="auto",
        attn_implementation=attn_impl,
        torch_dtype=(torch.bfloat16 if bf16 else "auto"),
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except Exception:
        kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tokenizer


def _chat_prompt(tokenizer: Any, user_prompt: str) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return f"user: {user_prompt}\nassistant:"


def _generate_text(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[str, int, float]:
    import torch

    text = _chat_prompt(tokenizer, prompt)
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = temperature > 0
    start = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - start

    gen_ids = out[0, inputs["input_ids"].shape[1] :]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return gen_text.strip(), int(gen_ids.shape[0]), elapsed


def _extract_mcq(row: Dict[str, Any], id_fields: Sequence[str]) -> Optional[Dict[str, Any]]:
    question = _extract_first_present(row, ["question", "query", "prompt", "input", "instruction"])
    if not question:
        return None

    choices = None
    if "choices" in row and isinstance(row["choices"], (list, tuple)) and len(row["choices"]) >= 2:
        choices = [str(c) for c in row["choices"]]
    elif "options" in row and isinstance(row["options"], (list, tuple)) and len(row["options"]) >= 2:
        choices = [str(c) for c in row["options"]]
    else:
        opt_keys = ["A", "B", "C", "D", "option_a", "option_b", "option_c", "option_d"]
        found = []
        for key in opt_keys:
            val = row.get(key)
            if val:
                found.append(str(val))
        if len(found) >= 2:
            choices = found

    if not choices:
        return None

    answer_raw = _extract_first_present(row, ["answer", "label", "target", "gold", "correct_answer"])
    if not answer_raw:
        return None

    answer = answer_raw.strip()
    if answer.isdigit():
        idx = int(answer)
        if 0 <= idx < len(choices):
            answer = chr(ord("A") + idx)
    answer = answer[:1].upper()
    if answer not in {"A", "B", "C", "D", "E"}:
        return None

    subject = _extract_first_present(row, ["subject", "category", "topic", "domain"]) or "unknown"
    sample_id = _extract_first_present(row, id_fields)
    return {
        "question": question,
        "choices": choices,
        "answer": answer,
        "subject": subject,
        "sample_id": sample_id,
    }


def _parse_choice(text: str) -> Optional[str]:
    m = re.search(r"\b([A-E])\b", text.upper())
    if m:
        return m.group(1)
    return None


def _choice_prompt(question: str, choices: List[str], fewshot: Sequence[Dict[str, Any]]) -> str:
    lines = ["Choose the correct option and answer with only A, B, C, or D."]

    for ex in fewshot:
        lines.append(f"Question: {ex['question']}")
        for i, c in enumerate(ex["choices"]):
            lines.append(f"{chr(ord('A') + i)}. {c}")
        lines.append(f"Answer: {ex['answer']}")
        lines.append("")

    lines.append(f"Question: {question}")
    for i, c in enumerate(choices):
        lines.append(f"{chr(ord('A') + i)}. {c}")
    lines.append("Answer:")
    return "\n".join(lines)


def _load_language_rows(dataset_id: str, split: str, lang: str, lang_field_candidates: Sequence[str], max_rows: Optional[int] = None):
    tried = []

    # Strategy 1: load split and filter by language field.
    try:
        ds = load_dataset(dataset_id, split=split)
        rows = []
        for row in ds:
            rlang = None
            for key in lang_field_candidates:
                if key in row:
                    rlang = normalize_lang(str(row[key]))
                    break
            if rlang == lang:
                rows.append(dict(row))
            if max_rows and len(rows) >= max_rows:
                break
        if rows:
            return rows
    except Exception as exc:
        tried.append(f"filter_split:{exc}")

    # Strategy 2: language config name.
    for name in [lang, lang.upper(), f"{lang}_test", f"{lang}_eval"]:
        try:
            ds = load_dataset(dataset_id, name=name, split=split)
            rows = [dict(row) for row in ds]
            if max_rows:
                rows = rows[:max_rows]
            if rows:
                return rows
        except Exception as exc:
            tried.append(f"config:{name}:{exc}")

    raise RuntimeError(f"Unable to load rows for {dataset_id} lang={lang}. Attempts: {tried[:3]}")


def evaluate_global_mmlu(
    model: Any,
    tokenizer: Any,
    cfg: Dict[str, Any],
    mode: str,
    model_label: str,
) -> Tuple[List[MetricRow], List[Dict[str, Any]]]:
    task_cfg = cfg["global_mmlu"][mode]
    dataset_id = task_cfg["dataset_id"]
    split = task_cfg.get("split", "test")
    langs = task_cfg.get("langs", ["ja", "ko", "en"])
    max_rows_per_lang = task_cfg.get("max_rows_per_lang")
    n_shot = int(task_cfg.get("n_shot", 5))
    matched_en_control = bool(task_cfg.get("matched_en_control", False))
    id_fields = task_cfg.get("id_fields", ["id", "question_id", "sample_id", "uid"])

    metrics: List[MetricRow] = []
    items: List[Dict[str, Any]] = []
    matched_id_pool: set[str] = set()

    for lang in langs:
        rows = _load_language_rows(
            dataset_id=dataset_id,
            split=split,
            lang=lang,
            lang_field_candidates=task_cfg.get("lang_fields", ["language", "lang"]),
            max_rows=max_rows_per_lang,
        )
        mcq_rows = [x for x in (_extract_mcq(r, id_fields) for r in rows) if x]
        split_label = lang

        if matched_en_control and lang in {"ja", "ko"}:
            matched_id_pool.update(str(x["sample_id"]) for x in mcq_rows if x.get("sample_id"))

        if matched_en_control and lang == "en" and matched_id_pool:
            filtered_rows = [x for x in mcq_rows if x.get("sample_id") and str(x["sample_id"]) in matched_id_pool]
            if filtered_rows:
                mcq_rows = filtered_rows
                split_label = "en_matched"

        if not mcq_rows:
            continue

        correct = 0
        total = 0
        token_count = 0
        elapsed_total = 0.0
        per_item = []

        for idx, sample in enumerate(tqdm(mcq_rows, desc=f"Global-MMLU {lang}", leave=False)):
            fewshot_pool = mcq_rows[:idx] + mcq_rows[idx + 1 :]
            fewshot = fewshot_pool[:n_shot]
            prompt = _choice_prompt(sample["question"], sample["choices"], fewshot)
            pred_text, gen_tokens, elapsed = _generate_text(
                model,
                tokenizer,
                prompt,
                max_new_tokens=int(task_cfg.get("max_new_tokens", 8)),
                temperature=float(task_cfg.get("temperature", 0.0)),
                top_p=float(task_cfg.get("top_p", 1.0)),
            )
            pred = _parse_choice(pred_text)
            is_correct = pred == sample["answer"]
            total += 1
            correct += int(is_correct)
            token_count += gen_tokens
            elapsed_total += elapsed
            per_item.append(
                {
                    "task": "global_mmlu",
                    "mode": mode,
                    "lang": lang,
                    "split_label": split_label,
                    "sample_id": sample.get("sample_id"),
                    "pred_text": pred_text,
                    "pred": pred,
                    "gold": sample["answer"],
                    "correct": is_correct,
                    "gen_tokens": gen_tokens,
                    "elapsed_sec": elapsed,
                }
            )

        acc = correct / max(total, 1)
        tps = token_count / max(elapsed_total, 1e-8)

        metrics.append(
            MetricRow(
                metric="global_mmlu_accuracy",
                value=acc,
                split=f"{mode}:{split_label}",
                model=model_label,
                extra={"n": total, "tokens_per_sec": tps, "matched_en_control": bool(matched_en_control and split_label == "en_matched")},
            )
        )
        items.extend(per_item)

    return metrics, items


def evaluate_flores(
    model: Any,
    tokenizer: Any,
    cfg: Dict[str, Any],
    mode: str,
    model_label: str,
) -> Tuple[List[MetricRow], List[Dict[str, Any]]]:
    task_cfg = cfg["flores"][mode]
    dataset_id = task_cfg["dataset_id"]
    split = task_cfg.get("split", "devtest")
    max_rows = task_cfg.get("max_rows")
    directions = [tuple(d) for d in task_cfg["directions"]]
    chrf = CHRF(word_order=2)

    ds = load_dataset(dataset_id, split=split)
    ds_columns = set(getattr(ds, "column_names", []))
    is_long_format = {"id", "iso_639_3", "text"}.issubset(ds_columns)

    metrics: List[MetricRow] = []
    items: List[Dict[str, Any]] = []

    long_rows: List[Dict[str, str]] = []
    if is_long_format:
        needed_langs = {normalize_lang(src) for src, _ in directions}
        needed_langs.update(normalize_lang(tgt) for _, tgt in directions)
        needed_langs.discard(None)

        by_id: Dict[str, Dict[str, str]] = {}
        complete_ids: List[str] = []

        for row in ds:
            rid = row.get("id")
            text = row.get("text")
            lang = normalize_lang(row.get("iso_639_3"))
            if rid is None or not text or lang not in needed_langs:
                continue

            sid = str(rid)
            bucket = by_id.setdefault(sid, {})
            bucket[lang] = str(text)
            if all(k in bucket for k in needed_langs):
                if not complete_ids or complete_ids[-1] != sid:
                    complete_ids.append(sid)
                if max_rows and len(complete_ids) >= int(max_rows):
                    break

        if complete_ids:
            long_rows = [by_id[sid] for sid in complete_ids if sid in by_id]
        else:
            for bucket in by_id.values():
                if all(k in bucket for k in needed_langs):
                    long_rows.append(bucket)
                    if max_rows and len(long_rows) >= int(max_rows):
                        break

    for src, tgt in directions:
        refs: List[str] = []
        hyps: List[str] = []
        token_count = 0
        elapsed_total = 0.0

        src_norm = normalize_lang(src) or src
        tgt_norm = normalize_lang(tgt) or tgt
        if is_long_format:
            selected_rows = long_rows[: int(max_rows)] if max_rows else long_rows
            flores_iter: Iterable[Dict[str, Any]] = selected_rows
        else:
            selected_rows = ds
            if max_rows:
                selected_rows = ds.select(range(min(int(max_rows), len(ds))))
            flores_iter = selected_rows

        for row in tqdm(flores_iter, desc=f"FLORES {src}->{tgt}", leave=False):
            if is_long_format:
                src_text = str(row.get(src_norm, "")).strip()
                tgt_text = str(row.get(tgt_norm, "")).strip()
                if not src_text or not tgt_text:
                    continue
            else:
                row = dict(row)
                src_text = _extract_first_present(row, FLORES_LANG_KEYS[src])
                tgt_text = _extract_first_present(row, FLORES_LANG_KEYS[tgt])
                if not src_text or not tgt_text:
                    continue

            prompt = (
                f"Translate the following text from {src} to {tgt}. "
                "Preserve named entities and formatting.\n\n"
                f"{src_text}"
            )
            pred_text, gen_tokens, elapsed = _generate_text(
                model,
                tokenizer,
                prompt,
                max_new_tokens=int(task_cfg.get("max_new_tokens", 256)),
                temperature=float(task_cfg.get("temperature", 0.0)),
                top_p=float(task_cfg.get("top_p", 1.0)),
            )

            refs.append(tgt_text)
            hyps.append(pred_text)
            token_count += gen_tokens
            elapsed_total += elapsed
            items.append(
                {
                    "task": "flores",
                    "mode": mode,
                    "direction": f"{src}->{tgt}",
                    "source": src_text,
                    "reference": tgt_text,
                    "prediction": pred_text,
                    "gen_tokens": gen_tokens,
                    "elapsed_sec": elapsed,
                }
            )

        if not refs:
            continue

        score = chrf.corpus_score(hyps, [refs]).score
        tps = token_count / max(elapsed_total, 1e-8)
        metrics.append(
            MetricRow(
                metric="flores_chrfpp",
                value=float(score),
                split=f"{mode}:{src}->{tgt}",
                model=model_label,
                extra={"n": len(refs), "tokens_per_sec": tps},
            )
        )

    return metrics, items


def evaluate_aya_eval(
    model: Any,
    tokenizer: Any,
    cfg: Dict[str, Any],
    mode: str,
    model_label: str,
) -> Tuple[List[MetricRow], List[Dict[str, Any]]]:
    task_cfg = cfg["aya_eval"][mode]
    dataset_id = task_cfg["dataset_id"]
    split = task_cfg.get("split", "test")
    name = task_cfg.get("name")
    name_fallbacks = list(task_cfg.get("name_fallbacks", []))
    if not name_fallbacks:
        name_fallbacks = ["aya_human_annotated", "dolly_human_edited", "dolly_machine_translated"]
    langs = set(task_cfg.get("langs", ["ja", "ko"]))

    ds = None
    tried: List[str] = []
    selected_name: Optional[str] = None
    candidates: List[Optional[str]] = [name] if name else [None]
    for candidate in name_fallbacks:
        if candidate not in candidates:
            candidates.append(candidate)
    for candidate in candidates:
        try:
            if candidate:
                cand_ds = load_dataset(dataset_id, name=candidate, split=split)
            else:
                cand_ds = load_dataset(dataset_id, split=split)

            has_target_lang = False
            for row in cand_ds:
                lang = normalize_lang(str(row.get("language") or row.get("lang") or ""))
                if lang in langs:
                    has_target_lang = True
                    break

            if has_target_lang:
                ds = cand_ds
                selected_name = candidate
                break
            tried.append(f"{candidate or '<default>'}: loaded_but_no_target_langs")
        except Exception as exc:
            tried.append(f"{candidate or '<default>'}: {exc}")
    if ds is None:
        raise RuntimeError(f"Unable to load {dataset_id} for aya_eval. Attempts: {tried[:4]}")
    max_rows_per_lang = int(task_cfg.get("max_rows_per_lang", 200))

    metrics: List[MetricRow] = []
    items: List[Dict[str, Any]] = []

    by_lang_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in ds:
        r = dict(row)
        lang = normalize_lang(str(r.get("language") or r.get("lang") or ""))
        if lang in langs and len(by_lang_rows[lang]) < max_rows_per_lang:
            by_lang_rows[lang].append(r)

    for lang, rows in by_lang_rows.items():
        confusion = 0
        total = 0
        tokens_total = 0
        elapsed_total = 0.0

        for row in tqdm(rows, desc=f"AyaEval {lang}", leave=False):
            prompt = _extract_first_present(row, ["inputs", "prompt", "instruction", "question", "text"])
            if not prompt:
                continue
            pred_text, gen_tokens, elapsed = _generate_text(
                model,
                tokenizer,
                prompt,
                max_new_tokens=int(task_cfg.get("max_new_tokens", 256)),
                temperature=float(task_cfg.get("temperature", 0.0)),
                top_p=float(task_cfg.get("top_p", 1.0)),
            )
            is_confused = language_confused(lang, pred_text)

            total += 1
            confusion += int(is_confused)
            tokens_total += gen_tokens
            elapsed_total += elapsed
            items.append(
                {
                    "task": "aya_eval",
                    "mode": mode,
                    "lang": lang,
                    "prompt": prompt,
                    "prediction": pred_text,
                    "confused": is_confused,
                    "detected_lang": detect_script_language(pred_text),
                    "gen_tokens": gen_tokens,
                    "elapsed_sec": elapsed,
                }
            )

        confusion_rate = confusion / max(total, 1)
        tps = tokens_total / max(elapsed_total, 1e-8)
        metrics.append(
            MetricRow(
                metric="language_confusion_rate",
                value=confusion_rate,
                split=f"{mode}:{lang}",
                model=model_label,
                extra={"n": total, "tokens_per_sec": tps, "aya_eval_config": selected_name or "<default>"},
            )
        )

    return metrics, items


def evaluate_entity_and_structured(
    model: Any,
    tokenizer: Any,
    cfg: Dict[str, Any],
    mode: str,
    model_label: str,
) -> Tuple[List[MetricRow], List[Dict[str, Any]]]:
    task_cfg = cfg["custom"][mode]

    metrics: List[MetricRow] = []
    items: List[Dict[str, Any]] = []

    # Entity preservation
    entity_correct = 0
    entity_total = 0
    entity_tokens = 0
    entity_elapsed = 0.0

    for sample in ENTITY_EVAL_SET:
        pred_text, gen_tokens, elapsed = _generate_text(
            model,
            tokenizer,
            sample["prompt"],
            max_new_tokens=int(task_cfg.get("entity_max_new_tokens", 96)),
            temperature=float(task_cfg.get("temperature", 0.0)),
            top_p=float(task_cfg.get("top_p", 1.0)),
        )
        ok = sample["entity"] in pred_text
        entity_correct += int(ok)
        entity_total += 1
        entity_tokens += gen_tokens
        entity_elapsed += elapsed
        items.append(
            {
                "task": "entity",
                "mode": mode,
                "lang": sample["lang"],
                "entity": sample["entity"],
                "prediction": pred_text,
                "correct": ok,
                "gen_tokens": gen_tokens,
                "elapsed_sec": elapsed,
            }
        )

    metrics.append(
        MetricRow(
            metric="entity_exact_rate",
            value=entity_correct / max(entity_total, 1),
            split=f"{mode}:entity",
            model=model_label,
            extra={"n": entity_total, "tokens_per_sec": entity_tokens / max(entity_elapsed, 1e-8)},
        )
    )

    # Structured output validity
    valid = 0
    total = 0
    struct_tokens = 0
    struct_elapsed = 0.0
    for sample in STRUCTURED_EVAL_SET:
        pred_text, gen_tokens, elapsed = _generate_text(
            model,
            tokenizer,
            sample["prompt"],
            max_new_tokens=int(task_cfg.get("structured_max_new_tokens", 128)),
            temperature=float(task_cfg.get("temperature", 0.0)),
            top_p=float(task_cfg.get("top_p", 1.0)),
        )

        json_obj = None
        try:
            match = re.search(r"\{.*\}", pred_text, flags=re.S)
            if match:
                json_obj = json.loads(match.group(0))
            else:
                json_obj = json.loads(pred_text)
        except Exception:
            json_obj = None

        is_valid = isinstance(json_obj, dict) and all(k in json_obj for k in sample["required_keys"])
        valid += int(is_valid)
        total += 1
        struct_tokens += gen_tokens
        struct_elapsed += elapsed
        items.append(
            {
                "task": "structured",
                "mode": mode,
                "lang": sample["lang"],
                "prediction": pred_text,
                "valid": is_valid,
                "gen_tokens": gen_tokens,
                "elapsed_sec": elapsed,
            }
        )

    metrics.append(
        MetricRow(
            metric="structured_valid_rate",
            value=valid / max(total, 1),
            split=f"{mode}:structured",
            model=model_label,
            extra={"n": total, "tokens_per_sec": struct_tokens / max(struct_elapsed, 1e-8)},
        )
    )

    return metrics, items


def _flatten_metrics(metric_rows: List[MetricRow]) -> List[Dict[str, Any]]:
    return [m.to_dict() for m in metric_rows]


def _build_eval_parity_manifest(cfg: Dict[str, Any], mode: str, tokenizer: Any, attn_implementation: str) -> Dict[str, Any]:
    gm = cfg.get("global_mmlu", {}).get(mode, {})
    flores = cfg.get("flores", {}).get(mode, {})
    aya_eval = cfg.get("aya_eval", {}).get(mode, {})
    custom = cfg.get("custom", {}).get(mode, {})

    def _gen_mode(temp: float) -> str:
        return "greedy" if float(temp) == 0.0 else "sampling"

    return {
        "mode": mode,
        "prompt_format": {
            "global_mmlu": "choice_prompt_v1",
            "flores": "translation_prompt_v1",
            "aya_eval": "single_turn_user_prompt",
            "custom_entity_structured": "single_turn_user_prompt",
        },
        "sampling": {
            "global_mmlu": {
                "generation_mode": _gen_mode(float(gm.get("temperature", 0.0))),
                "temperature": float(gm.get("temperature", 0.0)),
                "top_p": float(gm.get("top_p", 1.0)),
                "max_new_tokens": int(gm.get("max_new_tokens", 8)),
                "n_shot": int(gm.get("n_shot", 5)),
            },
            "flores": {
                "generation_mode": _gen_mode(float(flores.get("temperature", 0.0))),
                "temperature": float(flores.get("temperature", 0.0)),
                "top_p": float(flores.get("top_p", 1.0)),
                "max_new_tokens": int(flores.get("max_new_tokens", 256)),
            },
            "aya_eval": {
                "generation_mode": _gen_mode(float(aya_eval.get("temperature", 0.0))),
                "temperature": float(aya_eval.get("temperature", 0.0)),
                "top_p": float(aya_eval.get("top_p", 1.0)),
                "max_new_tokens": int(aya_eval.get("max_new_tokens", 256)),
            },
            "custom": {
                "generation_mode": _gen_mode(float(custom.get("temperature", 0.0))),
                "temperature": float(custom.get("temperature", 0.0)),
                "top_p": float(custom.get("top_p", 1.0)),
                "entity_max_new_tokens": int(custom.get("entity_max_new_tokens", 96)),
                "structured_max_new_tokens": int(custom.get("structured_max_new_tokens", 128)),
            },
        },
        "special_tokens": {
            "chat_template_present": bool(getattr(tokenizer, "chat_template", None)),
            "pad_token": tokenizer.pad_token,
            "pad_token_id": int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else None,
            "eos_token": tokenizer.eos_token,
            "eos_token_id": int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None,
        },
        "model_runtime": {
            "attn_implementation_arg": attn_implementation,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pre/post multilingual eval suite")
    parser.add_argument("--config", default="eval/configs/quick_8h.yaml")
    parser.add_argument("--mode", choices=["quick", "expanded"], required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-label", required=False, default=None)
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    return parser.parse_args()


def run_eval(
    config_path: str,
    mode: str,
    model_id: str,
    model_label: Optional[str],
    adapter_dir: Optional[str],
    output_dir: str,
    attn_implementation: str,
) -> Dict[str, Any]:
    cfg = load_yaml(config_path)
    model_label = model_label or model_id

    model, tokenizer = _load_model_tokenizer(model_id, adapter_dir, attn_implementation, bf16=True)

    all_metrics: List[MetricRow] = []
    all_items: List[Dict[str, Any]] = []

    for fn in [evaluate_global_mmlu, evaluate_flores, evaluate_aya_eval, evaluate_entity_and_structured]:
        metric_rows, items = fn(model, tokenizer, cfg, mode, model_label)
        all_metrics.extend(metric_rows)
        all_items.extend(items)

    out_dir = ensure_dir(output_dir)
    metrics_df = pd.DataFrame(_flatten_metrics(all_metrics))
    items_df = pd.DataFrame(all_items)

    metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    items_df.to_csv(out_dir / "items.csv", index=False)

    summary = {
        "completed_at_utc": now_utc_iso(),
        "mode": mode,
        "model_id": model_id,
        "model_label": model_label,
        "adapter_dir": adapter_dir,
        "n_metrics": len(metrics_df),
        "n_items": len(items_df),
        "eval_parity_manifest": _build_eval_parity_manifest(
            cfg=cfg,
            mode=mode,
            tokenizer=tokenizer,
            attn_implementation=attn_implementation,
        ),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    args = parse_args()
    summary = run_eval(
        config_path=args.config,
        mode=args.mode,
        model_id=args.model_id,
        model_label=args.model_label,
        adapter_dir=args.adapter_dir,
        output_dir=args.output_dir,
        attn_implementation=args.attn_implementation,
    )
    print(summary)


if __name__ == "__main__":
    main()
