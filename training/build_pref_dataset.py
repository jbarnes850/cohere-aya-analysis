#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset

from src.frontier_utils import (
    assert_no_benchmark_test_split,
    ensure_dir,
    load_yaml,
    normalize_lang,
    read_jsonl,
    timestamp_run_id,
    write_jsonl,
)


def _truncate_text(text: str) -> str:
    words = text.split()
    if len(words) > 6:
        return " ".join(words[: max(4, len(words) // 2)])
    return text[: max(8, len(text) // 2)]


def _break_json(text: str) -> str:
    t = text.strip()
    if t.startswith("{") and t.endswith("}"):
        return t[:-1]
    return text


def _remove_entities(text: str) -> str:
    text = re.sub(r"\b[A-Z][A-Za-z0-9&._-]+(?:\s+[A-Z][A-Za-z0-9&._-]+)*\b", "[ENTITY]", text)
    text = re.sub(r"[\u30a0-\u30ff]{2,}", "[ENTITY]", text)
    return text


def _language_confuse(lang: str, text: str) -> str:
    if lang == "ja":
        return "In English: " + _truncate_text(text)
    if lang == "ko":
        return "In English: " + _truncate_text(text)
    return "日本語で: " + _truncate_text(text)


def _extract_first_present(row: Dict[str, Any], candidates: Sequence[str]) -> Optional[str]:
    for key in candidates:
        if key in row and row[key] is not None:
            text = str(row[key]).strip()
            if text:
                return text
    return None


def _parse_choice(text: str) -> Optional[str]:
    match = re.search(r"\b([A-E])\b", str(text).upper())
    if match:
        return match.group(1)
    return None


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
            value = row.get(key)
            if value:
                found.append(str(value))
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

    return {
        "question": question,
        "choices": choices,
        "answer": answer,
        "sample_id": _extract_first_present(row, id_fields),
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


def _is_mcq_prompt(prompt: str) -> bool:
    option_lines = re.findall(r"(?m)^\s*[A-D][\.\)]\s+.+$", prompt)
    if len(option_lines) < 3:
        return False
    lower = prompt.lower()
    return ("answer:" in lower) or ("choose the correct option" in lower)


def _mcq_wrong_choice(chosen: str, rng: random.Random) -> str:
    gold = _parse_choice(chosen) or "A"
    candidates = [c for c in ["A", "B", "C", "D"] if c != gold]
    return rng.choice(candidates) if candidates else "A"


def _mcq_rejected(chosen: str, rng: random.Random) -> str:
    gold = _parse_choice(chosen) or chosen.strip()[:1].upper() or "A"
    wrong = _mcq_wrong_choice(chosen, rng)
    candidates = [
        wrong,
        f"The correct answer is {gold}.",
        f"{gold}. Because it best matches the question context.",
    ]
    return rng.choice(candidates)


def _make_rejected(lang: str, task: str, chosen: str, prompt: str, rng: random.Random) -> str:
    if _is_mcq_prompt(prompt):
        return _mcq_rejected(chosen, rng)

    candidates = [
        _truncate_text(chosen),
        _remove_entities(chosen),
        _language_confuse(lang, chosen),
    ]
    if task == "structured":
        candidates.append(_break_json(chosen))
    candidates = [c for c in candidates if c and c != chosen]
    return rng.choice(candidates) if candidates else _truncate_text(chosen)


def build_pref_rows(sft_rows: List[Dict[str, Any]], target_langs: List[str], seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    for row in sft_rows:
        lang = row.get("lang", "en")
        if lang not in target_langs and lang != "en":
            continue
        messages = row.get("messages") or []
        if not messages or messages[-1].get("role") != "assistant":
            continue

        prompt_turns = [m for m in messages if m.get("role") in {"system", "user"}]
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in prompt_turns)
        chosen = str(messages[-1].get("content", ""))
        if not prompt.strip() or not chosen.strip():
            continue

        task = str(row.get("task", "instruction"))
        rejected = _make_rejected(lang=lang, task=task, chosen=chosen, prompt=prompt, rng=rng)

        rows.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "lang": lang,
                "task": task,
                "source": row.get("source", "unknown"),
            }
        )

    rng.shuffle(rows)
    return rows


def _load_language_rows(
    dataset_id: str,
    split: str,
    lang: str,
    lang_field_candidates: Sequence[str],
    max_rows: Optional[int] = None,
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
            if max_rows and len(rows) >= max_rows:
                break
        if rows:
            return rows
    except Exception as exc:
        tried.append(f"filter_split:{exc}")

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


def _repeat_rows(rows: List[Dict[str, Any]], factor: int) -> List[Dict[str, Any]]:
    factor = max(1, int(factor))
    if factor == 1:
        return list(rows)
    out: List[Dict[str, Any]] = []
    for _ in range(factor):
        out.extend(dict(row) for row in rows)
    return out


def build_mcq_pref_rows(data_cfg: Dict[str, Any], seed: int, for_dev: bool = False) -> List[Dict[str, Any]]:
    mcq_cfg = data_cfg.get("mcq_preference", {})
    if not bool(mcq_cfg.get("enabled", True)):
        return []

    dataset_id = str(mcq_cfg.get("dataset_id", "CohereLabs/Global-MMLU-Lite"))
    split = str(mcq_cfg.get("split", "dev"))
    assert_no_benchmark_test_split(dataset_id=dataset_id, split=split, purpose="DPO MCQ preference construction")
    print(f"[pref] MCQ source verified for training-time use: dataset_id={dataset_id}, split={split}")
    langs_raw = mcq_cfg.get("langs", ["ja", "ko"])
    langs = [normalize_lang(str(lang)) for lang in langs_raw]
    langs = [lang for lang in langs if lang]
    if not langs:
        return []

    lang_fields = [str(x) for x in mcq_cfg.get("lang_field_candidates", ["language", "lang"])]
    id_fields = [str(x) for x in mcq_cfg.get("id_fields", ["id", "sample_id", "example_id", "uuid"])]
    max_choices = max(2, int(mcq_cfg.get("max_choices", 4)))
    fewshot_k = max(0, int(mcq_cfg.get("fewshot_k", 2)))
    max_rows_per_lang = max(64, int(mcq_cfg.get("max_rows_per_lang", 4096)))
    samples_key = "samples_per_lang_dev" if for_dev else "samples_per_lang_train"
    per_lang = max(1, int(mcq_cfg.get(samples_key, 64 if for_dev else 256)))
    repeat_key = "dev_oversample_factor" if for_dev else "train_oversample_factor"
    oversample_factor = max(1, int(mcq_cfg.get(repeat_key, 1)))

    rng = random.Random(seed + (101 if for_dev else 17))
    rows_out: List[Dict[str, Any]] = []

    for lang in langs:
        try:
            rows = _load_language_rows(
                dataset_id=dataset_id,
                split=split,
                lang=lang,
                lang_field_candidates=lang_fields,
                max_rows=max_rows_per_lang,
            )
        except Exception as exc:
            print(f"[pref] MCQ preference skipped for {lang}: {exc}")
            continue

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
        selected = parsed[fewshot_k : fewshot_k + per_lang] if len(parsed) > fewshot_k else parsed[:per_lang]

        for sample in selected:
            chosen = sample["answer"]
            prompt = "user: " + _choice_prompt(sample["question"], sample["choices"], fewshot)
            rejected = _mcq_rejected(chosen, rng)
            rows_out.append(
                {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "lang": lang,
                    "task": "mcq",
                    "source": "global_mmlu_lite",
                    "sample_id": sample.get("sample_id"),
                }
            )

    rng.shuffle(rows_out)
    return _repeat_rows(rows_out, oversample_factor)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build preference dataset (DPO) from SFT JSONL")
    parser.add_argument("--config", default="training/configs/tiny_aya_ja_ko_dpo.yaml")
    parser.add_argument("--sft-train-jsonl", required=True)
    parser.add_argument("--sft-dev-jsonl", required=False)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    run_id = args.run_id or timestamp_run_id(cfg.get("run_prefix", "tiny_aya_ja_ko_dpo"))
    output_dir = ensure_dir(Path(args.output_dir or cfg["paths"]["output_root"]) / run_id / "artifacts" / "pref_data")

    train_rows = read_jsonl(args.sft_train_jsonl)
    dev_rows = read_jsonl(args.sft_dev_jsonl) if args.sft_dev_jsonl else []

    data_cfg = cfg.get("data", {})
    target_langs = data_cfg.get("target_langs", ["ja", "ko"])
    seed = int(cfg.get("seed", 42))

    train_pref = build_pref_rows(train_rows, target_langs=target_langs, seed=seed)
    dev_pref = build_pref_rows(dev_rows, target_langs=target_langs, seed=seed + 1)
    train_mcq_pref = build_mcq_pref_rows(data_cfg=data_cfg, seed=seed + 7, for_dev=False)
    dev_mcq_pref = build_mcq_pref_rows(data_cfg=data_cfg, seed=seed + 11, for_dev=True)
    train_pref.extend(train_mcq_pref)
    dev_pref.extend(dev_mcq_pref)
    random.Random(seed + 99).shuffle(train_pref)
    random.Random(seed + 100).shuffle(dev_pref)

    write_jsonl(train_pref, output_dir / "train_pref.jsonl")
    write_jsonl(dev_pref, output_dir / "dev_pref.jsonl")

    print(f"Saved train preference rows: {len(train_pref)}")
    print(f"Saved dev preference rows: {len(dev_pref)}")
    print(f"MCQ train preference rows added: {len(train_mcq_pref)}")
    print(f"MCQ dev preference rows added: {len(dev_mcq_pref)}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
