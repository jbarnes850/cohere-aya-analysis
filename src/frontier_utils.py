from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml


LANG_ALIAS = {
    "en": "en",
    "eng": "en",
    "english": "en",
    "ja": "ja",
    "jpn": "ja",
    "jp": "ja",
    "japanese": "ja",
    "ko": "ko",
    "kor": "ko",
    "korean": "ko",
    "zh": "zh",
    "zho": "zh",
    "chi": "zh",
}

BENCHMARK_TRAINING_LEAKAGE_DATASETS = {
    "coherelabs/global-mmlu-lite",
    "coherelabs/global-mmlu",
    "coherelabs/aya_evaluation_suite",
    "openlanguagedata/flores_plus",
}

BENCHMARK_HELDOUT_SPLITS = {
    "test",
    "devtest",
}


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(data: Dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp_run_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: Iterable[Dict[str, Any]], path: str | Path) -> int:
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def normalize_lang(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text:
        return None
    if "_" in text:
        text = text.split("_", 1)[0]
    if "-" in text:
        text = text.split("-", 1)[0]
    return LANG_ALIAS.get(text, text)


def assert_no_benchmark_test_split(
    dataset_id: str,
    split: str,
    purpose: str,
    benchmark_datasets: Optional[Sequence[str]] = None,
    blocked_splits: Optional[Sequence[str]] = None,
) -> None:
    dataset_key = str(dataset_id).strip().lower()
    split_key = str(split).strip().lower()
    benchmark_set = {str(x).strip().lower() for x in (benchmark_datasets or BENCHMARK_TRAINING_LEAKAGE_DATASETS)}
    blocked_set = {str(x).strip().lower() for x in (blocked_splits or BENCHMARK_HELDOUT_SPLITS)}
    if dataset_key in benchmark_set and split_key in blocked_set:
        raise ValueError(
            f"Benchmark leakage guard triggered for {purpose}: "
            f"dataset_id={dataset_id}, split={split}. "
            "Use a non-heldout split (for example: dev) during training-time data selection."
        )


def _extract_messages_list(value: Any) -> Optional[List[Dict[str, str]]]:
    if not isinstance(value, list):
        return None
    out: List[Dict[str, str]] = []
    for turn in value:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role") or turn.get("from") or turn.get("speaker")
        content = turn.get("content") or turn.get("value") or turn.get("text")
        if role is None or content is None:
            continue
        role_l = str(role).strip().lower()
        if role_l in {"human", "user", "instruction", "prompt"}:
            role_l = "user"
        elif role_l in {"assistant", "model", "bot", "gpt"}:
            role_l = "assistant"
        elif role_l in {"system"}:
            role_l = "system"
        else:
            continue
        out.append({"role": role_l, "content": str(content)})
    return out if out else None


def coerce_messages(row: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    for key in ["messages", "conversations", "dialog", "chat"]:
        if key in row:
            parsed = _extract_messages_list(row[key])
            if parsed:
                user_idx = next((i for i, t in enumerate(parsed) if t["role"] == "user"), None)
                asst_idx = next((i for i, t in enumerate(parsed) if t["role"] == "assistant"), None)
                if user_idx is not None and asst_idx is not None and user_idx < asst_idx:
                    return parsed

    pairs: List[Tuple[Sequence[str], Sequence[str]]] = [
        (("instruction", "prompt", "input", "question", "query"), ("output", "response", "answer", "completion", "target")),
        (("inputs",), ("targets",)),
    ]

    for user_keys, asst_keys in pairs:
        user_val = next((row[k] for k in user_keys if k in row and row[k]), None)
        asst_val = next((row[k] for k in asst_keys if k in row and row[k]), None)
        if user_val is not None and asst_val is not None:
            return [
                {"role": "user", "content": str(user_val)},
                {"role": "assistant", "content": str(asst_val)},
            ]

    if "text" in row and isinstance(row["text"], str) and "\n\n" in row["text"]:
        user_text, asst_text = row["text"].split("\n\n", 1)
        if user_text.strip() and asst_text.strip():
            return [
                {"role": "user", "content": user_text.strip()},
                {"role": "assistant", "content": asst_text.strip()},
            ]

    if "text" in row and isinstance(row["text"], str):
        text = row["text"].strip()
        usr_m = re.search(r"<usr>\s*(.*?)\s*<bot>\s*(.*)$", text, flags=re.S)
        if usr_m:
            user_text = usr_m.group(1).strip()
            asst_text = usr_m.group(2).strip()
            if user_text and asst_text:
                return [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": asst_text},
                ]

        inst_m = re.search(r"\[INST\]\s*(.*?)\s*\[/INST\]\s*(.*)$", text, flags=re.S)
        if inst_m:
            user_text = inst_m.group(1).strip()
            asst_text = inst_m.group(2).strip()
            if user_text and asst_text:
                return [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": asst_text},
                ]

    return None


def coerce_lang(row: Dict[str, Any]) -> Optional[str]:
    def _norm(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            for item in value:
                n = _norm(item)
                if n:
                    return n
            return None
        if isinstance(value, dict):
            for key in ["lang", "language", "iso", "code", "target"]:
                if key in value:
                    n = _norm(value.get(key))
                    if n:
                        return n
            return None

        text = str(value).strip()
        n = normalize_lang(text)
        if n:
            return n

        # Handle serialized list-like values such as "['jpn']".
        tokens = re.findall(r"[A-Za-z]{2,}", text)
        for tok in tokens:
            n = normalize_lang(tok)
            if n:
                return n
        return None

    for key in ["lang", "language", "iso_lang", "locale", "target_language"]:
        if key in row:
            n = _norm(row.get(key))
            if n:
                return n
    for key in row.keys():
        if "lang" in key.lower() and row.get(key):
            n = _norm(row.get(key))
            if n:
                return n
    return None


def chat_to_text(tokenizer: Any, messages: List[Dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            pass
    chunks: List[str] = []
    for turn in messages:
        chunks.append(f"{turn['role']}: {turn['content']}")
    return "\n".join(chunks)


def safe_json_loads(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except Exception:
        return None


def detect_script_language(text: str) -> str:
    if not text:
        return "unknown"
    ja_hira_kata = len(re.findall(r"[\u3040-\u30ff]", text))
    ko_hangul = len(re.findall(r"[\uac00-\ud7af\u1100-\u11ff]", text))
    han = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin = len(re.findall(r"[A-Za-z]", text))

    if ko_hangul > max(ja_hira_kata, latin):
        return "ko"
    if ja_hira_kata > max(ko_hangul, latin):
        return "ja"
    if han > 0 and ja_hira_kata > 0:
        return "ja"
    if han > 0 and ko_hangul == 0 and ja_hira_kata == 0:
        return "zh"
    if latin > max(ja_hira_kata, ko_hangul, han):
        return "en"
    return "unknown"


def language_confused(expected_lang: str, generated_text: str) -> bool:
    observed = detect_script_language(generated_text)
    if expected_lang not in {"ja", "ko", "en", "zh"}:
        return False
    return observed not in {expected_lang, "unknown"}


def get_elapsed_minutes(start_time: float) -> float:
    return (time.time() - start_time) / 60.0


@dataclass
class MetricRow:
    metric: str
    value: float
    split: str
    model: str
    extra: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "metric": self.metric,
            "value": self.value,
            "split": self.split,
            "model": self.model,
        }
        if self.extra:
            data.update(self.extra)
        return data
