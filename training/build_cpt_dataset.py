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
    detect_script_language,
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


def _lang_matches_expected(expected: str, detected: str) -> bool:
    if detected == "unknown":
        return True
    if expected == "ja":
        # Kanji-heavy Japanese can be detected as zh.
        return detected in {"ja", "zh"}
    if expected == "ko":
        return detected == "ko"
    if expected == "en":
        return detected == "en"
    return True


def _quality_tokens(text: str, fallback_chars: int) -> List[str]:
    token_re = re.compile(r"[A-Za-z]+|[0-9]+|[\u3040-\u30ff]+|[\u3400-\u9fff]+|[\uac00-\ud7a3]+")
    tokens = [t.lower() for t in token_re.findall(text)]
    if len(tokens) >= 8:
        return tokens
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return []
    return list(compact[:fallback_chars])


def _assistant_quality_view(text: str) -> str:
    lines = text.splitlines()
    last_assistant_idx = -1
    for idx, line in enumerate(lines):
        if line.startswith("assistant:"):
            last_assistant_idx = idx
    if last_assistant_idx < 0:
        return text
    first_line = lines[last_assistant_idx][len("assistant:") :].strip()
    trailing = [ln.rstrip() for ln in lines[last_assistant_idx + 1 :]]
    return "\n".join([first_line] + trailing).strip()


def _simhash64(text: str, max_features: int) -> int:
    tokens = _quality_tokens(text, fallback_chars=max_features)
    if not tokens:
        return 0
    weights = [0] * 64
    for tok in tokens[:max_features]:
        hv = int.from_bytes(hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest(), "big")
        for bit in range(64):
            weights[bit] += 1 if ((hv >> bit) & 1) else -1
    fp = 0
    for bit, score in enumerate(weights):
        if score >= 0:
            fp |= 1 << bit
    return fp


def _hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _quality_score(text: str, lang: str, task: str, cfg: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    quality_cfg = cfg.get("data", {}).get("quality", {})
    max_repeated_line_ratio = float(quality_cfg.get("max_repeated_line_ratio", 0.45))
    max_repeated_char_run = max(4, int(quality_cfg.get("max_repeated_char_run", 8)))
    repetition_penalty = float(quality_cfg.get("repetition_penalty", 0.60))
    lang_mismatch_penalty = float(quality_cfg.get("lang_mismatch_penalty", 0.20))
    quality_text = _assistant_quality_view(text)
    normalized = re.sub(r"\s+", " ", quality_text).strip()
    if not normalized:
        return 0.0, {"lang_match": False, "detected_lang": "unknown"}

    lines = [ln.strip() for ln in quality_text.splitlines() if ln.strip()]
    kept_lines = [ln for ln in lines if len(ln) >= 10]
    line_counts = Counter(kept_lines)
    repeated_lines = sum(cnt for cnt in line_counts.values() if cnt > 1)
    repeated_line_ratio = (repeated_lines / len(kept_lines)) if kept_lines else 0.0

    has_long_char_repeat = bool(re.search(rf"(.)\1{{{max_repeated_char_run},}}", normalized))

    detected_lang = detect_script_language(normalized)
    lang_match = True
    if lang in {"ja", "ko", "en"} and task != "translation":
        lang_match = _lang_matches_expected(lang, detected_lang)

    score = 1.0
    repetition_bad = repeated_line_ratio > max_repeated_line_ratio or has_long_char_repeat
    if repetition_bad:
        score -= repetition_penalty
    if not lang_match:
        score -= lang_mismatch_penalty
    score = max(0.0, min(1.0, score))

    return score, {
        "lang_match": lang_match,
        "detected_lang": detected_lang,
        "repeated_line_ratio": repeated_line_ratio,
        "max_repeated_line_ratio": max_repeated_line_ratio,
        "max_repeated_char_run": max_repeated_char_run,
        "has_long_char_repeat": has_long_char_repeat,
        "repetition_bad": repetition_bad,
    }


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


def _quality_and_dedup_pool(items: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    data_cfg = cfg.get("data", {})
    quality_cfg = data_cfg.get("quality", {})
    near_dedup_cfg = data_cfg.get("near_dedup", {})

    enable_exact_dedup = bool(data_cfg.get("dedup", True))
    quality_enabled = bool(quality_cfg.get("enabled", True))
    drop_lang_mismatch = bool(quality_cfg.get("drop_lang_mismatch", quality_enabled))
    min_quality_score = float(quality_cfg.get("min_quality_score", 0.55))
    near_enabled = bool(near_dedup_cfg.get("enabled", True))
    hamming_threshold = max(0, int(near_dedup_cfg.get("hamming_threshold", 3)))
    band_bits = max(8, min(32, int(near_dedup_cfg.get("band_bits", 16))))
    max_bucket_candidates = max(8, int(near_dedup_cfg.get("max_bucket_candidates", 64)))
    simhash_max_features = max(64, int(near_dedup_cfg.get("simhash_max_features", 256)))

    out: List[Dict[str, Any]] = []
    seen_exact: set[str] = set()
    kept_fingerprints: List[int] = []
    band_index: Dict[Tuple[str, int, int], List[int]] = {}
    stats: Dict[str, Any] = {
        "input_rows": len(items),
        "kept_rows": 0,
        "dropped_duplicate_exact": 0,
        "dropped_duplicate_near": 0,
        "dropped_low_quality": 0,
        "dropped_lang_mismatch": 0,
        "quality_score_mean": 0.0,
        "quality_score_p10": 0.0,
        "quality_score_p90": 0.0,
    }
    quality_scores: List[float] = []

    for item in items:
        text = str(item.get("text", "")).strip()
        if not text:
            stats["dropped_low_quality"] += 1
            continue
        lang = normalize_lang(str(item.get("lang", ""))) or "unknown"
        task = str(item.get("task", "instruction"))

        score, details = _quality_score(text, lang=lang, task=task, cfg=cfg)
        quality_scores.append(score)
        is_lang_mismatch = quality_enabled and drop_lang_mismatch and (not bool(details["lang_match"]))
        is_low_quality = quality_enabled and score < min_quality_score
        if is_lang_mismatch:
            stats["dropped_lang_mismatch"] += 1
        if is_low_quality:
            stats["dropped_low_quality"] += 1
        if is_lang_mismatch or is_low_quality:
            continue

        if enable_exact_dedup:
            exact_hash = _hash_text(text)
            if exact_hash in seen_exact:
                stats["dropped_duplicate_exact"] += 1
                continue
            seen_exact.add(exact_hash)

        fp = 0
        if near_enabled:
            fp = _simhash64(text, max_features=simhash_max_features)
            candidate_ids: set[int] = set()
            for bit_offset in range(0, 64, band_bits):
                mask_width = min(band_bits, 64 - bit_offset)
                band_val = (fp >> bit_offset) & ((1 << mask_width) - 1)
                key = (lang, bit_offset, band_val)
                for idx in band_index.get(key, []):
                    candidate_ids.add(idx)

            near_duplicate = False
            for idx in candidate_ids:
                if _hamming64(fp, kept_fingerprints[idx]) <= hamming_threshold:
                    near_duplicate = True
                    break
            if near_duplicate:
                stats["dropped_duplicate_near"] += 1
                continue

        row = dict(item)
        row["quality_score"] = round(float(score), 6)
        out.append(row)
        if near_enabled:
            out_idx = len(out) - 1
            kept_fingerprints.append(fp)
            for bit_offset in range(0, 64, band_bits):
                mask_width = min(band_bits, 64 - bit_offset)
                band_val = (fp >> bit_offset) & ((1 << mask_width) - 1)
                key = (lang, bit_offset, band_val)
                bucket = band_index.setdefault(key, [])
                bucket.append(out_idx)
                if len(bucket) > max_bucket_candidates:
                    del bucket[0]

    if quality_scores:
        ordered = sorted(quality_scores)
        p10_idx = int((len(ordered) - 1) * 0.10)
        p90_idx = int((len(ordered) - 1) * 0.90)
        stats["quality_score_mean"] = round(sum(quality_scores) / len(quality_scores), 6)
        stats["quality_score_p10"] = round(ordered[p10_idx], 6)
        stats["quality_score_p90"] = round(ordered[p90_idx], 6)
    stats["kept_rows"] = len(out)
    return out, stats


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

    pool_quality: Dict[str, Dict[str, Any]] = {}
    for key in pools:
        pools[key], pool_quality[key] = _quality_and_dedup_pool(pools[key], cfg)

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
        "pool_quality": pool_quality,
        "quality_filters": {
            "enabled": bool(cfg.get("data", {}).get("quality", {}).get("enabled", True)),
            "min_quality_score": float(cfg.get("data", {}).get("quality", {}).get("min_quality_score", 0.55)),
            "drop_lang_mismatch": bool(cfg.get("data", {}).get("quality", {}).get("drop_lang_mismatch", True)),
            "near_dedup_enabled": bool(cfg.get("data", {}).get("near_dedup", {}).get("enabled", True)),
            "near_dedup_hamming_threshold": int(cfg.get("data", {}).get("near_dedup", {}).get("hamming_threshold", 3)),
            "near_dedup_band_bits": int(cfg.get("data", {}).get("near_dedup", {}).get("band_bits", 16)),
        },
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
