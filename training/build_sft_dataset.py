#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.frontier_utils import (
    coerce_lang,
    coerce_messages,
    detect_script_language,
    ensure_dir,
    load_yaml,
    normalize_lang,
    set_global_seed,
    timestamp_run_id,
    write_jsonl,
)


FLORES_LANG_KEYS = {
    "en": ["eng_Latn", "sentence_eng_Latn", "en", "english"],
    "ja": ["jpn_Jpan", "sentence_jpn_Jpan", "ja", "japanese"],
    "ko": ["kor_Hang", "sentence_kor_Hang", "ko", "korean"],
}

LANGUAGE_NAMES = {
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
}


ENTITY_EXAMPLES = [
    {
        "entity": "Fujitsu",
        "ja": "富士通",
        "ko": "후지쯔",
        "summary": "Global IT services and products company",
    },
    {
        "entity": "Toyota",
        "ja": "トヨタ",
        "ko": "토요타",
        "summary": "Automotive manufacturer",
    },
    {
        "entity": "Samsung Electronics",
        "ja": "サムスン電子",
        "ko": "삼성전자",
        "summary": "Electronics and semiconductor company",
    },
    {
        "entity": "LG Electronics",
        "ja": "LGエレクトロニクス",
        "ko": "LG전자",
        "summary": "Consumer electronics company",
    },
    {
        "entity": "Hyundai",
        "ja": "ヒュンダイ",
        "ko": "현대",
        "summary": "Automotive company",
    },
]


STRUCTURED_EXAMPLES = [
    {
        "lang": "ja",
        "prompt": "以下をJSONで返してください。company, revenue_2024_jpy, revenue_2025_jpy, yoy_growth_pct。会社名: 富士通。2024年売上: 500000000。2025年売上: 560000000",
        "answer": '{"company":"富士通","revenue_2024_jpy":500000000,"revenue_2025_jpy":560000000,"yoy_growth_pct":12.0}',
    },
    {
        "lang": "ko",
        "prompt": "다음을 JSON으로 반환하세요: company, revenue_2024_jpy, revenue_2025_jpy, yoy_growth_pct. 회사: 삼성전자, 2024 매출: 700000000, 2025 매출: 756000000",
        "answer": '{"company":"삼성전자","revenue_2024_jpy":700000000,"revenue_2025_jpy":756000000,"yoy_growth_pct":8.0}',
    },
    {
        "lang": "en",
        "prompt": "Return JSON with keys company, revenue_2024_jpy, revenue_2025_jpy, yoy_growth_pct. company=Toyota, revenue_2024_jpy=900000000, revenue_2025_jpy=981000000",
        "answer": '{"company":"Toyota","revenue_2024_jpy":900000000,"revenue_2025_jpy":981000000,"yoy_growth_pct":9.0}',
    },
]

SYNTH_TOPICS = [
    "cloud migration",
    "database indexing",
    "incident response",
    "model evaluation",
    "edge deployment",
    "data governance",
    "tokenization strategy",
    "latency optimization",
    "feature rollout",
    "query planning",
]

SYNTH_STYLE_KO = [
    "핵심만 간결하게",
    "실무적인 톤으로",
    "초보자도 이해할 수 있게",
    "전문가 관점에서",
    "리스크를 포함해",
]

SYNTH_STYLE_JA = [
    "簡潔に",
    "実務的な口調で",
    "初学者向けに",
    "専門家向けに",
    "リスクも含めて",
]


def _load_dataset_rows(
    dataset_path: str,
    split: str,
    max_examples: int,
    streaming: bool = True,
    dataset_name: Optional[str] = None,
    row_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> Iterable[Dict[str, Any]]:
    from datasets import load_dataset

    if max_examples <= 0:
        raise ValueError(f"max_examples must be > 0 for bounded dataset build; got {max_examples} for {dataset_path}:{split}")

    if dataset_name:
        ds = load_dataset(dataset_path, name=dataset_name, split=split, streaming=streaming)
    else:
        ds = load_dataset(dataset_path, split=split, streaming=streaming)
    count = 0
    for row in ds:
        row_dict = dict(row)
        if row_filter and not row_filter(row_dict):
            continue
        yield row_dict
        count += 1
        if max_examples and count >= max_examples:
            break


def _load_flores_rows(
    dataset_path: str,
    split: str,
    max_examples: int,
    streaming: bool = True,
    dataset_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    if max_examples <= 0:
        raise ValueError(f"max_examples must be > 0 for bounded dataset build; got {max_examples} for {dataset_path}:{split}")

    if dataset_name:
        ds = load_dataset(dataset_path, name=dataset_name, split=split, streaming=streaming)
    else:
        ds = load_dataset(dataset_path, split=split, streaming=streaming)

    rows: List[Dict[str, Any]] = []
    long_format_mode: Optional[bool] = None
    target_triplets = max(1, max_examples // 3)
    by_id: Dict[str, Dict[str, str]] = {}

    for row in ds:
        row_dict = dict(row)
        if long_format_mode is None:
            long_format_mode = {"id", "iso_639_3", "text"}.issubset(set(row_dict.keys()))

        if long_format_mode:
            rid = row_dict.get("id")
            iso = normalize_lang(row_dict.get("iso_639_3"))
            text = row_dict.get("text")
            if rid is None or iso not in {"en", "ja", "ko"} or not text:
                continue

            key = str(rid)
            bucket = by_id.setdefault(key, {})
            bucket[iso] = str(text)
            if {"en", "ja", "ko"}.issubset(set(bucket.keys())):
                rows.append(
                    {
                        "id": key,
                        "en": bucket["en"],
                        "ja": bucket["ja"],
                        "ko": bucket["ko"],
                    }
                )
                del by_id[key]
                if len(rows) >= target_triplets:
                    break
            continue

        # Wide-format fallback.
        text_by_lang = {lang: _extract_first_present(row_dict, keys) for lang, keys in FLORES_LANG_KEYS.items()}
        if all(text_by_lang.values()):
            rows.append(row_dict)
            if len(rows) >= max_examples:
                break

    return rows


def _extract_first_present(row: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    for key in candidates:
        if key in row and row[key]:
            return str(row[key])
        for row_key in row.keys():
            if row_key.lower() == key.lower() and row[row_key]:
                return str(row[row_key])
    return None


def _get_nested_value(row: Dict[str, Any], field_path: Optional[str]) -> Optional[Any]:
    if not field_path:
        return None
    cur: Any = row
    for key in str(field_path).split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _parse_allowed_pairs(raw_pairs: Any) -> set[Tuple[str, str]]:
    out: set[Tuple[str, str]] = set()
    if not isinstance(raw_pairs, list):
        return out
    for pair in raw_pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        src = normalize_lang(str(pair[0]))
        tgt = normalize_lang(str(pair[1]))
        if src and tgt:
            out.add((src, tgt))
    return out


def _collect_flores_eval_splits(eval_cfg: Dict[str, Any]) -> set[str]:
    flores_cfg = eval_cfg.get("flores", {})
    splits: set[str] = set()
    for mode in ["quick", "expanded"]:
        mode_cfg = flores_cfg.get(mode, {})
        dataset_id = str(mode_cfg.get("dataset_id", "")).strip().lower()
        split = str(mode_cfg.get("split", "")).strip()
        if dataset_id.endswith("flores_plus") and split:
            splits.add(split)
    return splits


def _validate_no_flores_split_overlap(
    sft_cfg: Dict[str, Any],
    quick_eval_cfg: Dict[str, Any],
    expanded_eval_cfg: Dict[str, Any],
) -> None:
    train_splits: set[str] = set()
    for ds_cfg in sft_cfg.get("datasets", {}).values():
        if not isinstance(ds_cfg, dict):
            continue
        dataset_path = str(ds_cfg.get("path", "")).strip().lower()
        split = str(ds_cfg.get("split", "")).strip()
        if dataset_path.endswith("flores_plus") and split:
            train_splits.add(split)

    if not train_splits:
        return

    eval_splits = _collect_flores_eval_splits(quick_eval_cfg) | _collect_flores_eval_splits(expanded_eval_cfg)
    overlap = sorted(train_splits & eval_splits)
    if overlap:
        raise RuntimeError(
            "FLORES split leakage detected between training and evaluation. "
            f"overlapping_splits={overlap}. "
            "Use a disjoint training split (for example train on dev and evaluate on devtest)."
        )


def _build_parallel_translation_rows(
    rows: Iterable[Dict[str, Any]],
    source: str,
    translation_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    src_field = str(translation_cfg.get("source_text_field", "")).strip()
    tgt_field = str(translation_cfg.get("target_text_field", "")).strip()
    src_lang = normalize_lang(str(translation_cfg.get("source_lang", "")))
    tgt_lang = normalize_lang(str(translation_cfg.get("target_lang", "")))
    bidirectional = bool(translation_cfg.get("bidirectional", True))
    preserve_entities = bool(translation_cfg.get("preserve_entities_hint", True))

    if not src_field or not tgt_field:
        raise ValueError(
            f"parallel_translation dataset {source} requires translation.source_text_field and translation.target_text_field"
        )
    if not src_lang or not tgt_lang:
        raise ValueError(f"parallel_translation dataset {source} requires translation.source_lang and translation.target_lang")

    directions = [(src_lang, tgt_lang)]
    if bidirectional and (src_lang != tgt_lang):
        directions.append((tgt_lang, src_lang))

    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        src_text_val = _get_nested_value(row, src_field)
        tgt_text_val = _get_nested_value(row, tgt_field)
        src_text = str(src_text_val).strip() if src_text_val is not None else ""
        tgt_text = str(tgt_text_val).strip() if tgt_text_val is not None else ""
        if not src_text or not tgt_text:
            continue

        text_by_lang = {src_lang: src_text, tgt_lang: tgt_text}
        for d_src, d_tgt in directions:
            src_name = LANGUAGE_NAMES.get(d_src, d_src.upper())
            tgt_name = LANGUAGE_NAMES.get(d_tgt, d_tgt.upper())
            preserve_line = "Preserve named entities exactly.\n\n" if preserve_entities else "\n\n"
            prompt = f"Translate the following text from {src_name} to {tgt_name}. {preserve_line}{text_by_lang[d_src]}"
            out.append(
                {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": text_by_lang[d_tgt]},
                    ],
                    "lang": d_tgt,
                    "task": "translation",
                    "bucket": "translation",
                    "source": source,
                    "translation_source_lang": d_src,
                    "translation_target_lang": d_tgt,
                    "sample_id": f"{source}:parallel:{idx:08d}:{d_src}->{d_tgt}",
                }
            )
    return out


def _normalize_translation_chat_rows(
    rows: Iterable[Dict[str, Any]],
    source: str,
    translation_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    def _nested_lang(row: Dict[str, Any], field_path: str) -> Optional[str]:
        val = _get_nested_value(row, field_path)
        return normalize_lang(str(val)) if val is not None else None

    src_field = translation_cfg.get("source_lang_field")
    tgt_field = translation_cfg.get("target_lang_field")
    allowed_pairs = _parse_allowed_pairs(translation_cfg.get("allowed_pairs", []))

    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        messages = coerce_messages(row)
        if not messages:
            continue

        src_lang = _nested_lang(row, str(src_field)) if src_field else None
        tgt_lang = _nested_lang(row, str(tgt_field)) if tgt_field else None

        if not src_lang:
            src_lang = _nested_lang(row, "source_lang")
        if not src_lang:
            src_lang = _nested_lang(row, "translation_source_lang")

        if not tgt_lang:
            tgt_lang = normalize_lang(coerce_lang(row))
        if not tgt_lang:
            tgt_lang = _nested_lang(row, "target_lang")
        if not tgt_lang:
            tgt_lang = _nested_lang(row, "translation_target_lang")
        if not tgt_lang:
            tgt_lang = detect_script_language(str(messages[-1].get("content", "")))

        if tgt_lang not in {"ja", "ko", "en"}:
            continue
        if allowed_pairs and (not src_lang or (src_lang, tgt_lang) not in allowed_pairs):
            continue

        out.append(
            {
                "messages": messages,
                "lang": tgt_lang,
                "task": "translation",
                "bucket": "translation",
                "source": source,
                "translation_source_lang": src_lang,
                "translation_target_lang": tgt_lang,
                "sample_id": f"{source}:chat_translation:{idx:08d}",
            }
        )
    return out


def _build_flores_translation_rows(rows: Iterable[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    directions = [
        ("en", "ja"),
        ("ja", "en"),
        ("en", "ko"),
        ("ko", "en"),
        ("ja", "ko"),
        ("ko", "ja"),
    ]
    names = {"en": "English", "ja": "Japanese", "ko": "Korean"}

    row_list = list(rows)
    if not row_list:
        return out

    # FLORES+ may appear as wide rows (all langs in one row) or long rows (one row per language with id/iso_639_3/text).
    long_format = {"id", "iso_639_3", "text"}.issubset(set(row_list[0].keys()))
    text_rows: List[Dict[str, str]] = []
    if long_format:
        grouped: Dict[str, Dict[str, str]] = {}
        for row in row_list:
            rid = row.get("id")
            iso = row.get("iso_639_3")
            text = row.get("text")
            if rid is None or not iso or not text:
                continue
            lang = normalize_lang(str(iso))
            if lang not in {"en", "ja", "ko"}:
                continue
            bucket = grouped.setdefault(str(rid), {})
            bucket[lang] = str(text)

        text_rows = [bucket for bucket in grouped.values() if {"en", "ja", "ko"}.issubset(set(bucket.keys()))]
    else:
        for row in row_list:
            text_by_lang: Dict[str, str] = {}
            for lang, keys in FLORES_LANG_KEYS.items():
                value = _extract_first_present(row, keys)
                if value:
                    text_by_lang[lang] = value
            if {"en", "ja", "ko"}.issubset(set(text_by_lang.keys())):
                text_rows.append(text_by_lang)

    for text_by_lang in text_rows:
        for src, tgt in directions:
            prompt = (
                f"Translate the following text from {names[src]} to {names[tgt]}. "
                "Preserve named entities exactly.\n\n"
                f"{text_by_lang[src]}"
            )
            out.append(
                {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": text_by_lang[tgt]},
                    ],
                    "lang": tgt,
                    "task": "translation",
                    "bucket": "translation",
                    "source": source,
                    "translation_source_lang": src,
                    "translation_target_lang": tgt,
                }
            )

    return out


def _build_entity_rows(n_rows: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    langs = ["ja", "ko", "en"]
    regions = ["APAC", "EMEA", "NA", "LATAM"]
    products = ["cloud platform", "edge device", "database service", "AI assistant", "chipset"]

    for i in range(n_rows):
        item = rng.choice(ENTITY_EXAMPLES)
        lang = rng.choice(langs)
        region = rng.choice(regions)
        product = rng.choice(products)
        revenue = rng.randint(120, 980)
        growth = rng.randint(-5, 25)
        if lang == "ja":
            prompt = (
                f"次の企業名をそのまま保持して日本語で2文を書いてください: {item['ja']}。"
                f"地域={region}、製品={product}、売上={revenue}億円、成長率={growth}% を必ず含めてください。"
            )
            answer = (
                f"{item['ja']}は{region}で{product}を展開する{item['summary']}です。"
                f"直近の売上は{revenue}億円で、前年同期比{growth}%でした。"
            )
            entity_key = item["ja"]
        elif lang == "ko":
            prompt = (
                f"다음 기업명을 그대로 유지해 한국어 2문장을 작성하세요: {item['ko']}. "
                f"지역={region}, 제품={product}, 매출={revenue}억 엔, 성장률={growth}%를 반드시 포함하세요."
            )
            answer = (
                f"{item['ko']}는 {region}에서 {product}를 운영하는 {item['summary']} 기업입니다. "
                f"최근 매출은 {revenue}억 엔이며 전년 대비 {growth}% 변동했습니다."
            )
            entity_key = item["ko"]
        else:
            prompt = (
                f"Write two English sentences and preserve this entity exactly: {item['entity']}. "
                f"Include region={region}, product={product}, revenue={revenue}00M JPY, growth={growth}%."
            )
            answer = (
                f"{item['entity']} is a {item['summary']} active in {region} with a focus on {product}. "
                f"It reported revenue of {revenue}00M JPY with {growth}% year-over-year change."
            )
            entity_key = item["entity"]

        rows.append(
            {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ],
                "lang": lang,
                "task": "entity",
                "bucket": "entity",
                "entity_key": entity_key,
                "source": "synthetic_entity",
                "sample_id": f"entity_{lang}_{i:07d}",
            }
        )
    return rows


def _build_structured_rows(n_rows: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    companies = [x["entity"] for x in ENTITY_EXAMPLES]
    for i in range(n_rows):
        sample = rng.choice(STRUCTURED_EXAMPLES)
        cpy = rng.choice(companies)
        rev_2024 = rng.randint(200_000_000, 1_600_000_000)
        delta = rng.randint(-120_000_000, 280_000_000)
        rev_2025 = max(50_000_000, rev_2024 + delta)
        yoy = round(((rev_2025 - rev_2024) / rev_2024) * 100.0, 2)
        uid = f"{sample['lang']}-{i:07d}"

        if sample["lang"] == "ja":
            prompt = (
                "以下をJSONで返してください。"
                "keys: company, revenue_2024_jpy, revenue_2025_jpy, yoy_growth_pct, report_id。"
                f"company={cpy}, revenue_2024_jpy={rev_2024}, revenue_2025_jpy={rev_2025}, report_id={uid}"
            )
        elif sample["lang"] == "ko":
            prompt = (
                "다음을 JSON으로 반환하세요. "
                "keys: company, revenue_2024_jpy, revenue_2025_jpy, yoy_growth_pct, report_id. "
                f"company={cpy}, revenue_2024_jpy={rev_2024}, revenue_2025_jpy={rev_2025}, report_id={uid}"
            )
        else:
            prompt = (
                "Return JSON with keys company, revenue_2024_jpy, revenue_2025_jpy, yoy_growth_pct, report_id. "
                f"company={cpy}, revenue_2024_jpy={rev_2024}, revenue_2025_jpy={rev_2025}, report_id={uid}"
            )
        answer = (
            "{"
            f"\"company\":\"{cpy}\","
            f"\"revenue_2024_jpy\":{rev_2024},"
            f"\"revenue_2025_jpy\":{rev_2025},"
            f"\"yoy_growth_pct\":{yoy},"
            f"\"report_id\":\"{uid}\""
            "}"
        )

        rows.append(
            {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ],
                "lang": sample["lang"],
                "task": "structured",
                "bucket": "structured",
                "required_keys": ["company", "revenue_2024_jpy", "revenue_2025_jpy", "yoy_growth_pct", "report_id"],
                "source": "synthetic_structured",
                "sample_id": f"structured_{sample['lang']}_{i:07d}",
            }
        )
    return rows


def _build_synthetic_instruction_rows(lang: str, n_rows: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    for i in range(n_rows):
        topic = rng.choice(SYNTH_TOPICS)
        k = rng.randint(3, 7)
        budget = rng.randint(50, 900)
        quarter = rng.choice(["Q1", "Q2", "Q3", "Q4"])
        uid = f"{lang}-inst-{i:07d}"

        if lang == "ko":
            style = rng.choice(SYNTH_STYLE_KO)
            prompt = (
                f"{topic} 프로젝트를 위한 실행 계획을 {k}개 항목으로 작성하세요. "
                f"{style}. 예산은 {budget}백만 엔, 기준 분기는 {quarter}, 식별자 {uid}를 포함하세요."
            )
            answer = (
                f"1) {quarter}에 {topic} 목표를 정렬하고 핵심 지표를 정의합니다.\n"
                f"2) 예산 {budget}백만 엔 범위에서 데이터/인프라 우선순위를 고정합니다.\n"
                f"3) 실험-검증 루프를 주 단위로 운영해 병목을 제거합니다.\n"
                f"4) 릴리스 전 품질 게이트와 롤백 조건을 문서화합니다.\n"
                f"5) 식별자 {uid} 기준으로 결과를 리뷰하고 다음 분기 액션을 확정합니다."
            )
        elif lang == "ja":
            style = rng.choice(SYNTH_STYLE_JA)
            prompt = (
                f"{topic}プロジェクトの実行計画を{k}項目で作成してください。"
                f"{style}。予算は{budget}百万円、対象四半期は{quarter}、識別子{uid}を含めてください。"
            )
            answer = (
                f"1) {quarter}の目標と{topic}のKPIを先に固定します。\n"
                f"2) 予算{budget}百万円の範囲でデータ基盤と運用優先度を定義します。\n"
                "3) 週次で実験と検証を回し、ボトルネックを解消します。\n"
                "4) リリース前に品質ゲートとロールバック条件を明文化します。\n"
                f"5) 識別子{uid}で成果をレビューし、次四半期の改善項目を確定します。"
            )
        else:
            prompt = (
                f"Write a {k}-point execution plan for a {topic} project. "
                f"Include budget={budget}M JPY, quarter={quarter}, id={uid}."
            )
            answer = (
                f"1) Align {topic} scope and KPIs for {quarter}.\n"
                f"2) Lock budget at {budget}M JPY across data, infra, and release tracks.\n"
                "3) Run weekly experiment/review cycles with clear ownership.\n"
                "4) Define quality gates and rollback policy before release.\n"
                f"5) Review outcomes under id={uid} and set next-step actions."
            )

        rows.append(
            {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ],
                "lang": lang,
                "task": "instruction",
                "bucket": f"{lang}_instruction",
                "source": f"synthetic_{lang}_instruction",
                "sample_id": uid,
            }
        )

    return rows


def _build_synthetic_translation_rows(n_rows: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    directions = [("en", "ja"), ("ja", "en"), ("en", "ko"), ("ko", "en"), ("ja", "ko"), ("ko", "ja")]
    names = {"en": "English", "ja": "Japanese", "ko": "Korean"}

    for i in range(n_rows):
        src, tgt = rng.choice(directions)
        topic = rng.choice(SYNTH_TOPICS)
        quarter = rng.choice(["Q1", "Q2", "Q3", "Q4"])
        team = rng.randint(1, 24)
        pct = rng.randint(3, 35)
        budget = rng.randint(60, 950)
        uid = f"syn-trans-{i:07d}"

        en_text = (
            f"In {quarter}, team {team} will improve {topic} by {pct}% with a budget of {budget}M JPY "
            f"(reference {uid})."
        )
        ja_text = (
            f"{quarter}にチーム{team}は{topic}を{pct}%改善し、予算{budget}百万円で実施します"
            f"（参照{uid}）。"
        )
        ko_text = (
            f"{quarter}에 팀{team}은 {topic}를 {pct}% 개선하고 예산 {budget}백만 엔으로 수행합니다"
            f"(참조 {uid})."
        )
        text_by_lang = {"en": en_text, "ja": ja_text, "ko": ko_text}

        prompt = (
            f"Translate the following text from {names[src]} to {names[tgt]}. "
            "Preserve numbers, identifiers, and named entities.\n\n"
            f"{text_by_lang[src]}"
        )
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": text_by_lang[tgt]},
                ],
                "lang": tgt,
                "task": "translation",
                "bucket": "translation",
                "source": "synthetic_translation",
                "sample_id": uid,
            }
        )

    return rows


def _extract_entity_candidates(text: str) -> List[str]:
    if not text:
        return []

    candidates: set[str] = set()
    for pattern in [
        r"\b[A-Z][A-Za-z0-9&._-]{2,}(?:\s+[A-Z][A-Za-z0-9&._-]{2,})*",
        r"[\u30a0-\u30ff]{2,}",
        r"[\uac00-\ud7af]{2,}",
        r"[\u4e00-\u9fff]{2,}",
    ]:
        for match in re.findall(pattern, text):
            token = str(match).strip().strip(".,;:!?()[]{}\"'")
            if len(token) >= 2:
                candidates.add(token)
    # Prefer longer entities first (e.g., "Samsung Electronics" over "Samsung").
    return sorted(candidates, key=lambda x: (-len(x), x))


def _build_entity_rows_from_source(
    source_rows: List[Dict[str, Any]],
    max_rows: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if max_rows <= 0:
        return []

    rng = random.Random(seed)
    candidates: List[Dict[str, Any]] = []
    for row in source_rows:
        messages = row.get("messages") or []
        if not isinstance(messages, list) or len(messages) < 2:
            continue
        if messages[-1].get("role") != "assistant":
            continue
        prompt_text = "\n".join(str(t.get("content", "")) for t in messages[:-1] if t.get("role") in {"user", "system"})
        answer_text = str(messages[-1].get("content", ""))
        if not prompt_text or not answer_text:
            continue

        shared = [x for x in _extract_entity_candidates(answer_text) if x in prompt_text or x in answer_text]
        if not shared:
            continue

        item = dict(row)
        item["task"] = "entity"
        item["bucket"] = "entity"
        item["entity_key"] = shared[0]
        item["source"] = f"{row.get('source', 'unknown')}:entity_mined"
        candidates.append(item)

    if len(candidates) <= max_rows:
        return candidates

    rng.shuffle(candidates)
    return candidates[:max_rows]


def _build_structured_rows_from_translation(
    translation_rows: List[Dict[str, Any]],
    max_rows: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if max_rows <= 0:
        return []

    rng = random.Random(seed)
    candidates: List[Dict[str, Any]] = []
    for idx, row in enumerate(translation_rows):
        messages = row.get("messages") or []
        if not isinstance(messages, list) or len(messages) < 2:
            continue
        if messages[-1].get("role") != "assistant":
            continue

        user_prompt = str(messages[0].get("content", "")).strip()
        target_text = str(messages[-1].get("content", "")).strip()
        if not user_prompt or not target_text:
            continue

        src_lang = normalize_lang(str(row.get("translation_source_lang", "unknown"))) or "unknown"
        tgt_lang = normalize_lang(str(row.get("translation_target_lang", row.get("lang", "unknown")))) or "unknown"
        prompt = (
            f"{user_prompt}\n\n"
            "Return only JSON with keys translation, source_lang, target_lang."
        )
        answer_obj = {
            "translation": target_text,
            "source_lang": src_lang,
            "target_lang": tgt_lang,
        }
        answer = json.dumps(answer_obj, ensure_ascii=False)
        candidates.append(
            {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ],
                "lang": tgt_lang,
                "task": "structured",
                "bucket": "structured",
                "required_keys": ["translation", "source_lang", "target_lang"],
                "source": f"{row.get('source', 'unknown')}:structured_json",
                "sample_id": f"structured_translation_{idx:08d}",
            }
        )

    if len(candidates) <= max_rows:
        return candidates

    rng.shuffle(candidates)
    return candidates[:max_rows]


def _bucket_targets_from_weights(total_examples: int, weights: Dict[str, float]) -> Dict[str, int]:
    targets = {bucket: int(round(total_examples * float(weight))) for bucket, weight in weights.items()}
    diff = int(total_examples) - sum(targets.values())
    if diff != 0 and targets:
        key = max(targets, key=lambda k: targets[k])
        targets[key] += diff
    return targets


def _resolve_bucket_targets(
    requested_total_examples: int,
    weights: Dict[str, float],
    available_counts: Dict[str, int],
    cfg: Dict[str, Any],
) -> Tuple[int, Dict[str, int]]:
    data_cfg = cfg.get("data", {})
    mode = str(data_cfg.get("target_mode", "auto_scale_to_available")).strip().lower()
    min_total_examples = int(data_cfg.get("min_total_examples", 1))

    if mode == "fixed":
        return requested_total_examples, _bucket_targets_from_weights(requested_total_examples, weights)

    total = int(requested_total_examples)
    while total > 0:
        targets = _bucket_targets_from_weights(total, weights)
        if all(int(targets.get(bucket, 0)) <= int(available_counts.get(bucket, 0)) for bucket in weights):
            if total < min_total_examples:
                break
            return total, targets
        total -= 1

    requested_targets = _bucket_targets_from_weights(requested_total_examples, weights)
    details = "\n".join(
        f"{bucket}: requested={requested_targets.get(bucket, 0)} available={available_counts.get(bucket, 0)}"
        for bucket in weights
    )
    raise RuntimeError(
        "Unable to satisfy source-backed bucket targets with current data supply.\n"
        f"requested_total_examples={requested_total_examples}, min_total_examples={min_total_examples}\n"
        f"{details}"
    )


def _sample_with_replacement(
    pool: List[Dict[str, Any]],
    n: int,
    rng: random.Random,
    with_replacement: bool = True,
) -> List[Dict[str, Any]]:
    if not pool or n <= 0:
        return []
    if with_replacement:
        if n <= len(pool):
            return rng.sample(pool, n)
        return [rng.choice(pool) for _ in range(n)]
    if n >= len(pool):
        copied = list(pool)
        rng.shuffle(copied)
        return copied
    return rng.sample(pool, n)


def _normalize_messages(rows: Iterable[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        messages = coerce_messages(row)
        if not messages:
            continue
        lang = coerce_lang(row)
        if lang is None:
            assistant_text = messages[-1]["content"] if messages else ""
            detected = detect_script_language(assistant_text)
            lang = detected if detected in {"ja", "ko", "en"} else "en"

        out.append(
            {
                "messages": messages,
                "lang": normalize_lang(lang) or "en",
                "task": "instruction",
                "source": source,
            }
        )
    return out


def _lang_matches_expected(expected: str, detected: str) -> bool:
    if detected == "unknown":
        return True
    if expected == "ja":
        # Kanji-only Japanese text can be detected as zh by script heuristic.
        return detected in {"ja", "zh"}
    if expected == "ko":
        return detected == "ko"
    if expected == "en":
        return detected == "en"
    return True


def _apply_quality_gates(rows: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    quality_cfg = cfg.get("data", {}).get("quality", {})
    max_chars_per_example = int(quality_cfg.get("max_chars_per_example", 16000))
    min_assistant_chars = int(quality_cfg.get("min_assistant_chars", 2))
    enable_dedup = bool(quality_cfg.get("dedup", True))
    drop_lang_mismatch = bool(quality_cfg.get("drop_lang_mismatch", True))

    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    stats = {
        "input_rows": len(rows),
        "kept_rows": 0,
        "dropped_malformed": 0,
        "dropped_lang_mismatch": 0,
        "dropped_duplicate": 0,
        "truncated_rows": 0,
    }

    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            stats["dropped_malformed"] += 1
            continue
        normalized_messages: List[Dict[str, str]] = []
        for turn in messages:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role", "")).strip()
            content = str(turn.get("content", "")).strip()
            if role and content:
                normalized_messages.append({"role": role, "content": content})
        if len(normalized_messages) < 2 or normalized_messages[-1]["role"] != "assistant":
            stats["dropped_malformed"] += 1
            continue
        if not any(t["role"] in {"user", "system"} for t in normalized_messages[:-1]):
            stats["dropped_malformed"] += 1
            continue

        total_chars = sum(len(t["content"]) for t in normalized_messages)
        if max_chars_per_example > 0 and total_chars > max_chars_per_example:
            keep_assistant = max_chars_per_example - sum(len(t["content"]) for t in normalized_messages[:-1])
            keep_assistant = max(keep_assistant, min_assistant_chars)
            normalized_messages[-1]["content"] = normalized_messages[-1]["content"][:keep_assistant].strip()
            if len(normalized_messages[-1]["content"]) < min_assistant_chars:
                stats["dropped_malformed"] += 1
                continue
            stats["truncated_rows"] += 1

        assistant_text = normalized_messages[-1]["content"]
        if len(assistant_text) < min_assistant_chars:
            stats["dropped_malformed"] += 1
            continue

        expected_lang = normalize_lang(row.get("lang")) or "en"
        task = str(row.get("task", "instruction"))
        if drop_lang_mismatch and expected_lang in {"ja", "ko", "en"} and task not in {"entity", "structured"}:
            detected = detect_script_language(assistant_text)
            if not _lang_matches_expected(expected_lang, detected):
                stats["dropped_lang_mismatch"] += 1
                continue

        row_copy = dict(row)
        row_copy["messages"] = normalized_messages
        row_copy["lang"] = expected_lang

        if enable_dedup:
            prompt_text = "\n".join(t["content"] for t in normalized_messages[:-1] if t["role"] in {"user", "system"})
            dedup_payload = {
                "lang": row_copy.get("lang"),
                "task": row_copy.get("task"),
                "prompt": prompt_text,
                "assistant": assistant_text,
            }
            dedup_key = hashlib.sha1(json.dumps(dedup_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
            if dedup_key in seen:
                stats["dropped_duplicate"] += 1
                continue
            seen.add(dedup_key)

        out.append(row_copy)

    stats["kept_rows"] = len(out)
    return out, stats


def _compute_post_qc_bucket_floor_report(
    rows: List[Dict[str, Any]],
    bucket_targets: Dict[str, int],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    quality_cfg = cfg.get("data", {}).get("quality", {})
    enforce = bool(quality_cfg.get("enforce_post_qc_bucket_floors", True))
    default_min_frac = float(quality_cfg.get("min_bucket_target_fraction", 0.50))
    per_bucket_min_frac = quality_cfg.get("min_bucket_target_fraction_by_bucket", {})

    critical_buckets_cfg = quality_cfg.get("critical_buckets")
    if isinstance(critical_buckets_cfg, list) and critical_buckets_cfg:
        critical_buckets = [str(x) for x in critical_buckets_cfg]
    else:
        critical_buckets = [k for k, v in bucket_targets.items() if int(v) > 0]

    bucket_counts = Counter(str(row.get("bucket", row.get("task", "unknown"))) for row in rows)
    checks: Dict[str, Dict[str, Any]] = {}
    failed: Dict[str, Dict[str, Any]] = {}

    for bucket in critical_buckets:
        target = int(bucket_targets.get(bucket, 0))
        if target <= 0:
            continue

        min_frac = float(per_bucket_min_frac.get(bucket, default_min_frac))
        min_required = int(math.ceil(target * min_frac))
        actual = int(bucket_counts.get(bucket, 0))
        ratio = float(actual / target) if target > 0 else 0.0
        status = "PASS" if actual >= min_required else "FAIL"

        payload = {
            "target": target,
            "actual": actual,
            "min_required": min_required,
            "min_fraction": min_frac,
            "actual_fraction_of_target": ratio,
            "status": status,
        }
        checks[bucket] = payload
        if status == "FAIL":
            failed[bucket] = payload

    return {
        "enforced": enforce,
        "default_min_fraction": default_min_frac,
        "critical_buckets": critical_buckets,
        "checks": checks,
        "failed": failed,
    }


def build_sft_dataset(cfg: Dict[str, Any], output_root: Path) -> Dict[str, Any]:
    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)
    rng = random.Random(seed)

    requested_total_examples = int(cfg["data"]["total_examples"])
    train_ratio = float(cfg["data"].get("train_ratio", 0.98))
    sampling_with_replacement = bool(cfg.get("data", {}).get("sampling_with_replacement", False))
    source_only = bool(cfg.get("data", {}).get("source_only", True))
    weights = {bucket: float(weight) for bucket, weight in cfg["data"]["weights"].items()}
    target_instruction_langs = {
        normalize_lang(str(x))
        for x in cfg.get("data", {}).get("target_instruction_langs", ["ja", "ko"])
        if normalize_lang(str(x))
    }

    def _instruction_lang_filter(row: Dict[str, Any]) -> bool:
        lang = normalize_lang(coerce_lang(row))
        if not lang:
            messages = coerce_messages(row)
            if messages:
                assistant_text = str(messages[-1].get("content", ""))
                detected = detect_script_language(assistant_text)
                lang = detected if detected in {"ja", "ko", "en"} else None
        return bool(lang and lang in target_instruction_langs)

    raw_pools: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    provisional_targets = _bucket_targets_from_weights(requested_total_examples, weights)
    min_pool_multiplier = float(cfg.get("data", {}).get("source_derivation_pool_multiplier", 2.0))

    dataset_cfg = cfg["datasets"]
    for name, ds_cfg in dataset_cfg.items():
        task_mode = str(ds_cfg.get("task", "instruction")).strip().lower()
        if task_mode not in {"instruction", "translation_chat", "parallel_translation"}:
            raise ValueError(f"Unsupported dataset task={task_mode} for {name}")

        path = ds_cfg["path"]
        split = ds_cfg.get("split", "train")
        streaming = bool(ds_cfg.get("streaming", True))
        row_filter: Optional[Callable[[Dict[str, Any]], bool]] = None
        apply_lang_filter = bool(ds_cfg.get("apply_instruction_lang_filter", name in {"aya_dataset", "aya_collection"}))
        if apply_lang_filter and task_mode == "instruction":
            row_filter = _instruction_lang_filter

        dataset_names: List[Optional[str]]
        names_cfg = ds_cfg.get("names")
        if isinstance(names_cfg, list) and names_cfg:
            dataset_names = [str(x) for x in names_cfg]
        elif ds_cfg.get("name"):
            dataset_names = [str(ds_cfg["name"])]
        else:
            dataset_names = [None]

        per_name_max_cfg = ds_cfg.get("max_examples_per_name")
        default_max_examples = int(ds_cfg.get("max_examples", 0))

        for dataset_name in dataset_names:
            max_examples = default_max_examples
            if isinstance(per_name_max_cfg, dict) and dataset_name and dataset_name in per_name_max_cfg:
                max_examples = int(per_name_max_cfg[dataset_name])
            elif isinstance(per_name_max_cfg, int):
                max_examples = int(per_name_max_cfg)
            if max_examples <= 0:
                continue

            source_label = f"{name}:{dataset_name}" if dataset_name else name
            print(f"Loading {source_label} from {path} [{split}] ...")
            try:
                if name == "flores_plus":
                    rows = _load_flores_rows(
                        dataset_path=path,
                        split=split,
                        max_examples=max_examples,
                        streaming=streaming,
                        dataset_name=dataset_name,
                    )
                else:
                    rows = list(
                        _load_dataset_rows(
                            path,
                            split=split,
                            max_examples=max_examples,
                            streaming=streaming,
                            dataset_name=dataset_name,
                            row_filter=row_filter,
                        )
                    )
            except Exception as exc:
                print(f"WARN: failed to load {source_label}: {exc}")
                continue

            if name == "flores_plus":
                extracted = _build_flores_translation_rows(rows, source=source_label)
                raw_pools["translation"].extend(extracted)
                continue

            if task_mode == "parallel_translation":
                extracted = _build_parallel_translation_rows(
                    rows=rows,
                    source=source_label,
                    translation_cfg=ds_cfg.get("translation", {}),
                )
                raw_pools["translation"].extend(extracted)
                continue

            if task_mode == "translation_chat":
                extracted = _normalize_translation_chat_rows(
                    rows=rows,
                    source=source_label,
                    translation_cfg=ds_cfg.get("translation", {}),
                )
                raw_pools["translation"].extend(extracted)
                continue

            normalized = _normalize_messages(rows, source=source_label)
            for item in normalized:
                if item["lang"] == "ja":
                    item["bucket"] = "ja_instruction"
                    raw_pools["ja_instruction"].append(item)
                elif item["lang"] == "ko":
                    item["bucket"] = "ko_instruction"
                    raw_pools["ko_instruction"].append(item)
                elif item["lang"] == "en":
                    item["bucket"] = "en_instruction"
                    raw_pools["en_instruction"].append(item)

    entity_target_pool = int(math.ceil(provisional_targets.get("entity", 0) * min_pool_multiplier))
    structured_target_pool = int(math.ceil(provisional_targets.get("structured", 0) * min_pool_multiplier))

    entity_source_rows: List[Dict[str, Any]] = []
    entity_source_rows.extend(raw_pools.get("ja_instruction", []))
    entity_source_rows.extend(raw_pools.get("ko_instruction", []))
    entity_source_rows.extend(raw_pools.get("translation", []))
    raw_pools["entity"].extend(
        _build_entity_rows_from_source(
            source_rows=entity_source_rows,
            max_rows=max(entity_target_pool, 1),
            seed=seed + 17,
        )
    )
    raw_pools["structured"].extend(
        _build_structured_rows_from_translation(
            translation_rows=raw_pools.get("translation", []),
            max_rows=max(structured_target_pool, 1),
            seed=seed + 23,
        )
    )

    aug_cfg = cfg.get("data", {}).get("synthetic_augmentation", {})
    if bool(aug_cfg.get("enabled", False)):
        aug_min_pool_multiplier = float(aug_cfg.get("min_pool_multiplier", 1.5))
        instruction_buckets = aug_cfg.get("instruction_buckets", ["ko_instruction"])
        for bucket in instruction_buckets:
            if not isinstance(bucket, str) or not bucket.endswith("_instruction"):
                continue
            lang = bucket.replace("_instruction", "")
            target = int(provisional_targets.get(bucket, 0))
            if target <= 0:
                continue
            required_pool = int(math.ceil(target * aug_min_pool_multiplier))
            current_pool = len(raw_pools.get(bucket, []))
            if current_pool >= required_pool:
                continue
            add_n = required_pool - current_pool
            raw_pools[bucket].extend(_build_synthetic_instruction_rows(lang=lang, n_rows=add_n, seed=seed + 101))

        if bool(aug_cfg.get("augment_translation", False)):
            translation_target = int(provisional_targets.get("translation", 0))
            if translation_target > 0:
                required_pool = int(math.ceil(translation_target * aug_min_pool_multiplier))
                current_pool = len(raw_pools.get("translation", []))
                if current_pool < required_pool:
                    add_n = required_pool - current_pool
                    raw_pools["translation"].extend(_build_synthetic_translation_rows(n_rows=add_n, seed=seed + 131))
        if bool(aug_cfg.get("augment_entity_structured", False)):
            entity_target = int(provisional_targets.get("entity", 0))
            struct_target = int(provisional_targets.get("structured", 0))
            if len(raw_pools.get("entity", [])) < entity_target:
                add_n = entity_target - len(raw_pools.get("entity", []))
                raw_pools["entity"].extend(_build_entity_rows(add_n, seed + 151))
            if len(raw_pools.get("structured", [])) < struct_target:
                add_n = struct_target - len(raw_pools.get("structured", []))
                raw_pools["structured"].extend(_build_structured_rows(add_n, seed + 181))

    pool_quality_stats: Dict[str, Dict[str, int]] = {}
    for bucket in weights.keys():
        cleaned, stats = _apply_quality_gates(raw_pools.get(bucket, []), cfg)
        raw_pools[bucket] = cleaned
        pool_quality_stats[bucket] = stats

    available_counts = {bucket: len(raw_pools.get(bucket, [])) for bucket in weights.keys()}
    total_examples, bucket_targets = _resolve_bucket_targets(
        requested_total_examples=requested_total_examples,
        weights=weights,
        available_counts=available_counts,
        cfg=cfg,
    )

    final_rows: List[Dict[str, Any]] = []
    for bucket, target_n in bucket_targets.items():
        sampled = _sample_with_replacement(
            raw_pools.get(bucket, []),
            target_n,
            rng,
            with_replacement=sampling_with_replacement,
        )
        final_rows.extend(sampled)

    if not final_rows:
        raise RuntimeError("No rows remained after quality gates; check dataset quality settings.")

    quality_stats = {
        "input_rows": len(final_rows),
        "kept_rows": len(final_rows),
        "dropped_malformed": 0,
        "dropped_lang_mismatch": 0,
        "dropped_duplicate": 0,
        "truncated_rows": 0,
    }

    if source_only:
        bad_sources = sorted({str(row.get("source", "")) for row in final_rows if str(row.get("source", "")).startswith("synthetic_")})
        if bad_sources:
            raise RuntimeError(
                "Source-only mode is enabled, but synthetic rows were selected.\n"
                + "\n".join(bad_sources)
            )

    floor_report = _compute_post_qc_bucket_floor_report(final_rows, bucket_targets, cfg)
    if floor_report["enforced"] and floor_report["failed"]:
        failures = []
        for bucket, payload in floor_report["failed"].items():
            failures.append(
                f"{bucket}: actual={payload['actual']} target={payload['target']} "
                f"(required>={payload['min_required']}, frac={payload['actual_fraction_of_target']:.3f})"
            )
        raise RuntimeError(
            "Post-QC bucket floor check failed. Refusing to continue with skewed mixture.\n"
            + "\n".join(failures)
        )

    rng.shuffle(final_rows)

    split_idx = int(len(final_rows) * train_ratio)
    train_rows = final_rows[:split_idx]
    dev_rows = final_rows[split_idx:]

    output_root = ensure_dir(output_root)
    n_train = write_jsonl(train_rows, output_root / "train.jsonl")
    n_dev = write_jsonl(dev_rows, output_root / "dev.jsonl")

    bucket_counts = Counter(str(row.get("bucket", row.get("task", "unknown"))) for row in final_rows)
    task_counts = Counter(str(row.get("task", "unknown")) for row in final_rows)
    lang_counts = Counter(row["lang"] for row in final_rows)
    source_counts = Counter(row["source"] for row in final_rows)

    meta = {
        "seed": seed,
        "requested_total_examples": requested_total_examples,
        "resolved_total_examples": total_examples,
        "total_examples": len(final_rows),
        "train_examples": n_train,
        "dev_examples": n_dev,
        "bucket_targets": bucket_targets,
        "available_counts": available_counts,
        "bucket_counts": dict(bucket_counts),
        "task_counts": dict(task_counts),
        "lang_counts": dict(lang_counts),
        "source_counts": dict(source_counts),
        "pool_quality": pool_quality_stats,
        "quality": quality_stats,
        "post_qc_bucket_floor_report": floor_report,
    }

    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SFT dataset for Tiny Aya JA/KO frontier training")
    parser.add_argument("--config", default="training/configs/tiny_aya_ja_ko_sft.yaml")
    parser.add_argument("--output-dir", default=None, help="Override output dir")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--quick-eval-config", default="eval/configs/quick_8h.yaml")
    parser.add_argument("--expanded-eval-config", default="eval/configs/expanded_frontier.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    quick_eval_cfg = load_yaml(args.quick_eval_config)
    expanded_eval_cfg = load_yaml(args.expanded_eval_config)
    _validate_no_flores_split_overlap(cfg, quick_eval_cfg, expanded_eval_cfg)

    run_id = args.run_id or timestamp_run_id(cfg.get("run_prefix", "tiny_aya_ja_ko"))
    base_output = Path(args.output_dir or cfg["paths"]["output_root"]) / run_id / "artifacts" / "data"
    ensure_dir(base_output)

    meta = build_sft_dataset(cfg, base_output)

    from src.frontier_utils import dump_yaml

    dump_yaml(meta, base_output / "dataset_meta.yaml")
    print(f"Saved dataset to {base_output}")
    print(meta)


if __name__ == "__main__":
    main()
