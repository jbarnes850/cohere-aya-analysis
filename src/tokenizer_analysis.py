"""Tokenizer analysis for Aya Expanse 8B."""

import logging
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from src.prompts import (
    ENTERPRISE_ENTITIES,
    LANGUAGE_NAMES,
    MIXED_SCRIPT_STRINGS,
)

logger = logging.getLogger(__name__)

CHAR_BASED_LANGUAGES = {"ja", "ko", "zh", "th"}
WORD_BASED_LANGUAGES = {"en", "vi", "id"}

MODEL_ID = "CohereLabs/aya-expanse-8b"


def load_tokenizer(model_id: str = MODEL_ID) -> AutoTokenizer:
    """Load the tokenizer."""
    logger.info("Loading tokenizer: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return tokenizer


def compute_fertility(
    tokenizer: AutoTokenizer,
    text: str,
    language: str,
) -> Dict[str, float]:
    """Compute token fertility for a single text."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    n_tokens = len(tokens)

    if language in CHAR_BASED_LANGUAGES:
        chars = re.sub(r"\s+", "", text)
        n_units = len(chars)
        metric = "tokens_per_char"
    else:
        words = text.split()
        n_units = len(words)
        metric = "tokens_per_word"

    fertility = n_tokens / max(n_units, 1)
    return {
        "n_tokens": n_tokens,
        "n_units": n_units,
        "fertility": fertility,
        "metric": metric,
    }


def compute_byte_fallback_rate(
    tokenizer: AutoTokenizer,
    text: str,
) -> float:
    """Compute fraction of byte-level fallback tokens."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    n_byte = 0
    for tid in token_ids:
        tok_str = tokenizer.convert_ids_to_tokens(tid)
        if tok_str is None:
            continue
        if re.match(r"<0x[0-9A-Fa-f]{2}>", tok_str):
            n_byte += 1
        elif len(tok_str) == 1 and ord(tok_str) < 32:
            n_byte += 1
    return n_byte / max(len(token_ids), 1)


def analyze_named_entity_splitting(
    tokenizer: AutoTokenizer,
    entities: Dict[str, List[str]],
) -> pd.DataFrame:
    """Analyze how entities split across languages."""
    rows = []
    for lang, names in entities.items():
        for name in names:
            token_ids = tokenizer.encode(name, add_special_tokens=False)
            token_strs = [tokenizer.convert_ids_to_tokens(tid) for tid in token_ids]
            rows.append({
                "language": lang,
                "entity": name,
                "n_tokens": len(token_ids),
                "tokens": token_strs,
                "token_ids": token_ids,
            })
    return pd.DataFrame(rows)


def analyze_script_mixing(
    tokenizer: AutoTokenizer,
    mixed_strings: List[str],
) -> pd.DataFrame:
    """Analyze tokenizer behavior on mixed-script text."""
    rows = []
    for text in mixed_strings:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        token_strs = [tokenizer.convert_ids_to_tokens(tid) for tid in token_ids]

        ascii_count = 0
        cjk_count = 0
        other_count = 0
        for tok in token_strs:
            if tok is None:
                continue
            clean = tok.replace("▁", "").replace("Ġ", "")
            if not clean:
                continue
            if all(ord(c) < 128 for c in clean):
                ascii_count += 1
            elif any("\u4e00" <= c <= "\u9fff" or "\u3040" <= c <= "\u30ff"
                     or "\uac00" <= c <= "\ud7af" or "\u0e00" <= c <= "\u0e7f"
                     for c in clean):
                cjk_count += 1
            else:
                other_count += 1

        rows.append({
            "text": text,
            "n_tokens": len(token_ids),
            "n_ascii_tokens": ascii_count,
            "n_cjk_tokens": cjk_count,
            "n_other_tokens": other_count,
            "tokens": token_strs,
        })
    return pd.DataFrame(rows)


def load_japanese_dataset_samples(n_samples: int = 1000, seed: int = 42) -> List[str]:
    """Load random samples from izumi-lab/llm-japanese-dataset."""
    from datasets import load_dataset

    logger.info("Loading izumi-lab/llm-japanese-dataset (sampling %d)", n_samples)
    ds = load_dataset("izumi-lab/llm-japanese-dataset", split="train", streaming=True)

    rng = np.random.RandomState(seed)
    samples = []
    for i, example in enumerate(ds):
        text = example.get("text") or example.get("output") or example.get("input", "")
        if not text or len(text.strip()) < 10:
            continue
        if len(samples) < n_samples:
            samples.append(text.strip())
        else:
            j = rng.randint(0, i + 1)
            if j < n_samples:
                samples[j] = text.strip()
        if i >= n_samples * 20:
            break
    logger.info("Loaded %d Japanese text samples", len(samples))
    return samples


def load_flores_samples(
    n_per_lang: int = 1000,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Load FLORES+ devtest samples with a fallback."""
    from datasets import load_dataset

    FLORES_LANG_MAP = {
        "en": "eng_Latn",
        "ko": "kor_Hang",
        "zh": "cmn_Hans",
        "vi": "vie_Latn",
        "id": "ind_Latn",
        "th": "tha_Thai",
    }

    rng = np.random.RandomState(seed)
    result = {}

    for lang, flores_code in FLORES_LANG_MAP.items():
        try:
            logger.info("Loading FLORES+ devtest for %s (%s)", lang, flores_code)
            ds = load_dataset(
                "openlanguagedata/flores_plus",
                flores_code,
                split="devtest",
            )
            texts = [row["text"] for row in ds if row.get("text") and len(row["text"].strip()) > 10]
            if len(texts) > n_per_lang:
                indices = rng.choice(len(texts), n_per_lang, replace=False)
                texts = [texts[i] for i in indices]
            result[lang] = texts
            logger.info("  Loaded %d samples for %s", len(texts), lang)
        except Exception as e:
            logger.warning("Failed to load FLORES+ for %s: %s. Using fallback.", lang, e)
            result[lang] = _fallback_samples(lang, n_per_lang)

    return result


def _fallback_samples(lang: str, n_per_lang: int) -> List[str]:
    """Fallback samples when FLORES+ is unavailable."""
    fallbacks = {
        "en": [
            "The quarterly financial report indicates strong growth in the Asia-Pacific region.",
            "Machine learning models require careful hyperparameter tuning for optimal performance.",
            "The board of directors approved the merger with a unanimous vote.",
            "Customer satisfaction surveys reveal a significant improvement in service quality.",
            "The new API endpoint supports both REST and GraphQL query formats.",
        ],
        "ko": [
            "분기별 재무 보고서는 아시아 태평양 지역에서 강한 성장을 보여줍니다.",
            "기계 학습 모델은 최적의 성능을 위해 신중한 하이퍼파라미터 조정이 필요합니다.",
            "이사회는 만장일치로 합병을 승인했습니다.",
            "고객 만족도 조사 결과 서비스 품질이 크게 향상된 것으로 나타났습니다.",
            "새로운 API 엔드포인트는 REST와 GraphQL 쿼리 형식을 모두 지원합니다.",
        ],
        "zh": [
            "季度财务报告显示亚太地区增长强劲。",
            "机器学习模型需要仔细的超参数调优以获得最佳性能。",
            "董事会以全票通过了合并决议。",
            "客户满意度调查显示服务质量有显著提升。",
            "新的API端点支持REST和GraphQL两种查询格式。",
        ],
        "vi": [
            "Bao cao tai chinh hang quy cho thay tang truong manh me tai khu vuc chau A Thai Binh Duong.",
            "Cac mo hinh hoc may doi hoi tinh chinh sieu tham so can than de dat hieu suat toi uu.",
            "Hoi dong quan tri da nhat tri phe duyet viec sap nhap.",
            "Khao sat muc do hai long cua khach hang cho thay chat luong dich vu duoc cai thien dang ke.",
            "Diem cuoi API moi ho tro ca dinh dang truy van REST va GraphQL.",
        ],
        "id": [
            "Laporan keuangan kuartalan menunjukkan pertumbuhan kuat di kawasan Asia-Pasifik.",
            "Model pembelajaran mesin memerlukan penyesuaian hyperparameter yang cermat untuk kinerja optimal.",
            "Dewan direksi menyetujui merger dengan suara bulat.",
            "Survei kepuasan pelanggan mengungkapkan peningkatan signifikan dalam kualitas layanan.",
            "Endpoint API baru mendukung format kueri REST dan GraphQL.",
        ],
        "th": [
            "รายงานการเงินรายไตรมาสแสดงการเติบโตอย่างแข็งแกร่งในภูมิภาคเอเชียแปซิฟิก",
            "โมเดลการเรียนรู้ของเครื่องต้องการการปรับไฮเปอร์พารามิเตอร์อย่างระมัดระวังเพื่อประสิทธิภาพที่ดีที่สุด",
            "คณะกรรมการบริษัทอนุมัติการควบรวมกิจการด้วยมติเอกฉันท์",
            "การสำรวจความพึงพอใจของลูกค้าเผยให้เห็นการปรับปรุงคุณภาพบริการอย่างมีนัยสำคัญ",
            "จุดปลายทาง API ใหม่รองรับรูปแบบการค้นหาทั้ง REST และ GraphQL",
        ],
    }
    texts = fallbacks.get(lang, ["Sample text."] * 5)
    return (texts * ((n_per_lang // len(texts)) + 1))[:n_per_lang]


def run_tokenizer_analysis(
    output_dir: str = "outputs",
    model_id: str = MODEL_ID,
    n_ja_samples: int = 1000,
    n_other_samples: int = 100,
    use_japanese_dataset: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Run tokenizer analysis."""
    output_path = Path(output_dir)
    (output_path / "tables").mkdir(parents=True, exist_ok=True)
    (output_path / "figures").mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(model_id)
    logger.info("Tokenizer vocab size: %d", tokenizer.vocab_size)

    fertility_rows = []

    if use_japanese_dataset:
        try:
            ja_texts = load_japanese_dataset_samples(n_ja_samples)
        except Exception as e:
            logger.warning("Failed to load JA dataset: %s. Using curated samples.", e)
            ja_texts = load_flores_samples(n_other_samples).get("ja", [])
            if not ja_texts:
                ja_texts = ["日本語のテスト文です。"] * n_other_samples
    else:
        ja_texts = ["日本語のテスト文です。"] * n_other_samples

    for text in ja_texts:
        result = compute_fertility(tokenizer, text, "ja")
        result["language"] = "ja"
        result["byte_fallback_rate"] = compute_byte_fallback_rate(tokenizer, text)
        fertility_rows.append(result)

    other_samples = load_flores_samples(n_other_samples)
    for lang, texts in other_samples.items():
        for text in texts:
            result = compute_fertility(tokenizer, text, lang)
            result["language"] = lang
            result["byte_fallback_rate"] = compute_byte_fallback_rate(tokenizer, text)
            fertility_rows.append(result)

    fertility_df = pd.DataFrame(fertility_rows)

    fertility_stats = fertility_df.groupby("language").agg(
        mean_fertility=("fertility", "mean"),
        std_fertility=("fertility", "std"),
        median_fertility=("fertility", "median"),
        mean_byte_fallback=("byte_fallback_rate", "mean"),
        n_samples=("fertility", "count"),
        metric=("metric", "first"),
    ).reset_index()
    fertility_stats["language_name"] = fertility_stats["language"].map(LANGUAGE_NAMES)

    fertility_stats.to_csv(output_path / "tables" / "tokenizer_fertility.csv", index=False)
    logger.info("Saved tokenizer fertility table")

    entity_df = analyze_named_entity_splitting(tokenizer, ENTERPRISE_ENTITIES)
    entity_df.to_csv(output_path / "tables" / "entity_splitting.csv", index=False)
    logger.info("Saved entity splitting table")

    mixing_df = analyze_script_mixing(tokenizer, MIXED_SCRIPT_STRINGS)
    mixing_df.to_csv(output_path / "tables" / "script_mixing.csv", index=False)
    logger.info("Saved script mixing table")

    print("\n=== Tokenizer Fertility Analysis (Aya Expanse 8B) ===")
    print(fertility_stats[["language_name", "mean_fertility", "std_fertility",
                           "mean_byte_fallback", "metric", "n_samples"]].to_string(index=False))

    print("\n=== CLT Paper Comparison (Harrasse et al. Table 6) ===")
    print("Arabic fertility: 1.97, English fertility: 1.53")
    print("Arabic coherence: 0.10, English coherence: 0.42")

    return {
        "fertility": fertility_stats,
        "fertility_raw": fertility_df,
        "entities": entity_df,
        "script_mixing": mixing_df,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_tokenizer_analysis()
