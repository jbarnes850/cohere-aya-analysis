#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset

from src.frontier_utils import (
    assert_no_benchmark_test_split,
    dump_yaml,
    ensure_dir,
    load_yaml,
    normalize_lang,
    read_jsonl,
    set_global_seed,
    timestamp_run_id,
    write_jsonl,
)


def _extract_first_present(row: Dict[str, Any], candidates: Sequence[str]) -> Optional[str]:
    for key in candidates:
        if key in row and row[key] is not None:
            text = str(row[key]).strip()
            if text:
                return text
    return None


def _render_messages(messages: Sequence[Dict[str, Any]]) -> str:
    chunks: List[str] = []
    for turn in messages:
        role = str(turn.get("role", "")).strip()
        content = str(turn.get("content", "")).strip()
        if role and content:
            chunks.append(f"{role}: {content}")
    return "\n".join(chunks)


def _clean_text(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text.strip())


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extract_mcq(row: Dict[str, Any], id_fields: Sequence[str]) -> Optional[Dict[str, Any]]:
    question = _extract_first_present(row, ["question", "query", "prompt", "input", "instruction"])
    if not question:
        return None

    choices: Optional[List[str]] = None
    if "choices" in row and isinstance(row["choices"], (list, tuple)) and len(row["choices"]) >= 2:
        choices = [str(c) for c in row["choices"]]
    elif "options" in row and isinstance(row["options"], (list, tuple)) and len(row["options"]) >= 2:
        choices = [str(c) for c in row["options"]]
    else:
        opt_keys = ["A", "B", "C", "D", "option_a", "option_b", "option_c", "option_d"]
        found: List[str] = []
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

    sample_id = _extract_first_present(row, id_fields)
    return {
        "question": question,
        "choices": choices,
        "answer": answer,
        "sample_id": sample_id,
    }


def _choice_prompt(question: str, choices: List[str], fewshot: Sequence[Dict[str, Any]]) -> str:
    lines = ["Choose the correct option and answer with only A, B, C, or D."]
    for ex in fewshot:
        lines.append(f"Question: {ex['question']}")
        for idx, choice in enumerate(ex["choices"]):
            lines.append(f"{chr(ord('A') + idx)}. {choice}")
        lines.append(f"Answer: {ex['answer']}")
        lines.append("")
    lines.append(f"Question: {question}")
    for idx, choice in enumerate(choices):
        lines.append(f"{chr(ord('A') + idx)}. {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


def _load_language_rows(
    dataset_id: str,
    split: str,
    lang: str,
    lang_field_candidates: Sequence[str],
    max_rows: int,
) -> List[Dict[str, Any]]:
    tried: List[str] = []
    try:
        ds = load_dataset(dataset_id, split=split)
        rows: List[Dict[str, Any]] = []
        for row in ds:
            row_lang: Optional[str] = None
            for key in lang_field_candidates:
                if key in row:
                    row_lang = normalize_lang(str(row[key]))
                    break
            if row_lang == lang:
                rows.append(dict(row))
            if len(rows) >= max_rows:
                break
        if rows:
            return rows
    except Exception as exc:
        tried.append(f"filter_split:{exc}")

    for name in [lang, lang.upper(), f"{lang}_test", f"{lang}_eval"]:
        try:
            ds = load_dataset(dataset_id, name=name, split=split)
            rows = [dict(row) for row in ds]
            if rows:
                return rows[:max_rows]
        except Exception as exc:
            tried.append(f"config:{name}:{exc}")

    raise RuntimeError(f"Unable to load rows for {dataset_id} lang={lang}. Attempts: {tried[:3]}")


def _build_targets(total: int, weights: Dict[str, float]) -> Dict[str, int]:
    raw = {k: total * float(v) for k, v in weights.items()}
    out = {k: int(math.floor(v)) for k, v in raw.items()}
    assigned = sum(out.values())
    if assigned < total:
        remainder = total - assigned
        frac_order = sorted(raw.keys(), key=lambda k: raw[k] - out[k], reverse=True)
        for key in frac_order:
            if remainder <= 0:
                break
            out[key] += 1
            remainder -= 1
    return out


def _redistribute_to_capacity(targets: Dict[str, int], available: Dict[str, int]) -> Dict[str, int]:
    adjusted = {k: min(int(targets.get(k, 0)), int(available.get(k, 0))) for k in targets}
    deficit = sum(int(targets.get(k, 0)) - adjusted[k] for k in targets)
    if deficit <= 0:
        return adjusted
    spare = {k: max(0, int(available.get(k, 0)) - adjusted[k]) for k in targets}
    for key in sorted(spare.keys(), key=lambda k: spare[k], reverse=True):
        if deficit <= 0:
            break
        take = min(deficit, spare[key])
        adjusted[key] += take
        deficit -= take
    return adjusted


def _append_pool_item(
    pools: Dict[str, List[Dict[str, Any]]],
    bucket: str,
    text: str,
    lang: str,
    task: str,
    source: str,
    sample_id: Optional[str],
    cfg: Dict[str, Any],
) -> None:
    min_chars = int(cfg["data"].get("min_chars", 16))
    max_chars = int(cfg["data"].get("max_chars", 16000))
    cleaned = _clean_text(text)
    if len(cleaned) < min_chars or len(cleaned) > max_chars:
        return
    pools[bucket].append(
        {
            "text": cleaned,
            "lang": lang,
            "task": task,
            "source": source,
            "bucket": bucket,
            "sample_id": sample_id,
        }
    )


def _dedup_pool(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        h = _hash_text(item["text"])
        if h in seen:
            continue
        seen.add(h)
        out.append(item)
    return out


def build_cpt_dataset(
    cfg: Dict[str, Any],
    sft_train_rows: List[Dict[str, Any]],
    sft_dev_rows: List[Dict[str, Any]],
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(seed)
    pools: Dict[str, List[Dict[str, Any]]] = {
        "ja_knowledge": [],
        "ko_knowledge": [],
        "ja_ko_instruction_mcq": [],
        "en_replay": [],
        "helper_transfer": [],
    }

    all_rows = list(sft_train_rows) + list(sft_dev_rows)
    for row in all_rows:
        messages = row.get("messages") or []
        if not isinstance(messages, list) or not messages:
            continue
        text = _render_messages(messages)
        lang = normalize_lang(str(row.get("lang", ""))) or "unknown"
        task = str(row.get("task", "instruction"))
        source = str(row.get("source", "sft_pool"))
        sample_id = str(row.get("sample_id", "")).strip() or None

        if lang == "ja" and task != "translation":
            _append_pool_item(pools, "ja_knowledge", text, lang, task, source, sample_id, cfg)
        if lang == "ko" and task != "translation":
            _append_pool_item(pools, "ko_knowledge", text, lang, task, source, sample_id, cfg)
        if lang == "en":
            _append_pool_item(pools, "en_replay", text, lang, task, source, sample_id, cfg)
        if task == "translation":
            _append_pool_item(pools, "helper_transfer", text, lang, task, source, sample_id, cfg)
        if lang in {"ja", "ko"} and task in {"instruction", "entity", "structured"}:
            _append_pool_item(pools, "ja_ko_instruction_mcq", text, lang, task, source, sample_id, cfg)

    mcq_cfg = cfg["data"].get("mcq_dev", {})
    if bool(mcq_cfg.get("enabled", True)):
        dataset_id = str(mcq_cfg.get("dataset_id", "CohereLabs/Global-MMLU-Lite"))
        split = str(mcq_cfg.get("split", "dev"))
        assert_no_benchmark_test_split(dataset_id=dataset_id, split=split, purpose="CPT MCQ dev augmentation")
        langs = [normalize_lang(str(x)) for x in mcq_cfg.get("langs", ["ja", "ko"])]
        langs = [x for x in langs if x]
        lang_fields = [str(x) for x in mcq_cfg.get("lang_field_candidates", ["language", "lang"])]
        id_fields = [str(x) for x in mcq_cfg.get("id_fields", ["id", "sample_id", "example_id", "uuid"])]
        max_rows_per_lang = max(64, int(mcq_cfg.get("max_rows_per_lang", 1024)))
        rows_per_lang = max(16, int(mcq_cfg.get("rows_per_lang", 384)))
        fewshot_k = max(0, int(mcq_cfg.get("fewshot_k", 2)))
        max_choices = max(2, int(mcq_cfg.get("max_choices", 4)))

        for lang in langs:
            rows = _load_language_rows(
                dataset_id=dataset_id,
                split=split,
                lang=lang,
                lang_field_candidates=lang_fields,
                max_rows=max_rows_per_lang,
            )
            parsed: List[Dict[str, Any]] = []
            for row in rows:
                sample = _extract_mcq(row, id_fields=id_fields)
                if not sample:
                    continue
                if len(sample["choices"]) > max_choices:
                    continue
                parsed.append(sample)
            if not parsed:
                continue
            rng.shuffle(parsed)
            fewshot = parsed[:fewshot_k]
            selected = parsed[fewshot_k : fewshot_k + rows_per_lang] if len(parsed) > fewshot_k else parsed[:rows_per_lang]
            for sample in selected:
                prompt = _choice_prompt(sample["question"], sample["choices"], fewshot)
                text = f"user: {prompt}\nassistant: {sample['answer']}"
                _append_pool_item(
                    pools,
                    "ja_ko_instruction_mcq",
                    text,
                    lang,
                    "mcq",
                    "global_mmlu_lite_dev",
                    sample.get("sample_id"),
                    cfg,
                )

    if bool(cfg["data"].get("dedup", True)):
        for key in pools:
            pools[key] = _dedup_pool(pools[key])

    weights = {k: float(v) for k, v in cfg["data"]["bucket_weights"].items()}
    total_requested = int(cfg["data"].get("total_texts", 120000))
    min_total = int(cfg["data"].get("min_total_texts", 60000))
    available = {k: len(v) for k, v in pools.items()}
    initial_targets = _build_targets(total_requested, weights)
    targets = _redistribute_to_capacity(initial_targets, available)
    resolved_total = sum(targets.values())
    if resolved_total < min_total:
        raise RuntimeError(
            f"CPT dataset too small after capacity checks: resolved_total={resolved_total}, min_total={min_total}, available={available}"
        )

    selected: List[Dict[str, Any]] = []
    for bucket, target in targets.items():
        candidates = list(pools.get(bucket, []))
        rng.shuffle(candidates)
        selected.extend(candidates[:target])
    rng.shuffle(selected)

    train_ratio = float(cfg["data"].get("train_ratio", 0.98))
    split_idx = int(len(selected) * train_ratio)
    train_rows = selected[:split_idx]
    dev_rows = selected[split_idx:]

    meta = {
        "seed": seed,
        "requested_total_texts": total_requested,
        "resolved_total_texts": len(selected),
        "train_texts": len(train_rows),
        "dev_texts": len(dev_rows),
        "bucket_weights": weights,
        "bucket_targets": targets,
        "bucket_available": available,
        "bucket_counts": dict(Counter(x.get("bucket", "unknown") for x in selected)),
        "lang_counts": dict(Counter(x.get("lang", "unknown") for x in selected)),
        "source_counts": dict(Counter(x.get("source", "unknown") for x in selected)),
        "leakage_guard": {
            "benchmark_test_split_allowed_for_training": False,
        },
    }
    return train_rows, dev_rows, meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CPT text dataset from SFT pools + MCQ dev augmentation")
    parser.add_argument("--config", default="training/configs/tiny_aya_ja_ko_cpt.yaml")
    parser.add_argument("--sft-train-jsonl", required=True)
    parser.add_argument("--sft-dev-jsonl", required=False)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)

    run_id = args.run_id or timestamp_run_id(cfg.get("run_prefix", "tiny_aya_ja_ko_cpt"))
    output_dir = ensure_dir(Path(args.output_dir or cfg["paths"]["output_root"]) / run_id / "artifacts" / "cpt_data")

    sft_train_rows = read_jsonl(args.sft_train_jsonl)
    sft_dev_rows = read_jsonl(args.sft_dev_jsonl) if args.sft_dev_jsonl else []
    train_rows, dev_rows, meta = build_cpt_dataset(cfg, sft_train_rows=sft_train_rows, sft_dev_rows=sft_dev_rows, seed=seed)

    write_jsonl(train_rows, output_dir / "train_text.jsonl")
    write_jsonl(dev_rows, output_dir / "dev_text.jsonl")
    dump_yaml(meta, output_dir / "cpt_meta.yaml")

    print(f"Saved CPT train texts: {len(train_rows)}")
    print(f"Saved CPT dev texts: {len(dev_rows)}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
