#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import random
import re
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback

from src.frontier_utils import (
    assert_no_benchmark_test_split,
    dump_yaml,
    ensure_dir,
    language_confused,
    load_yaml,
    now_utc_iso,
    normalize_lang,
    set_global_seed,
    timestamp_run_id,
)


def _build_lora_config(cfg: Dict[str, Any], qlora_mode: bool = False) -> LoraConfig:
    lora_cfg = cfg["lora"]
    rank = int(lora_cfg["r"])
    alpha = int(lora_cfg["lora_alpha"])
    if qlora_mode:
        rank = int(cfg["fallback"]["qlora"]["r"])
        alpha = int(cfg["fallback"]["qlora"]["lora_alpha"])
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=float(lora_cfg["lora_dropout"]),
        bias=str(lora_cfg.get("bias", "none")),
        task_type="CAUSAL_LM",
        target_modules=list(lora_cfg["target_modules"]),
        layers_to_transform=list(lora_cfg["layers_to_transform"]),
        modules_to_save=list(lora_cfg.get("modules_to_save", [])),
        layers_pattern=lora_cfg.get("layers_pattern", "layers"),
    )


def _load_model(cfg: Dict[str, Any], qlora_mode: bool = False) -> AutoModelForCausalLM:
    model_id = cfg["model"]["base_model"]
    attn_impl = str(cfg["model"].get("attn_implementation", "flash_attention_2"))
    if attn_impl == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
        # Keep training moving when FA2 wheel is unavailable on the target image.
        print("flash_attn not installed; falling back to attn_implementation=sdpa")
        attn_impl = "sdpa"
    use_bf16 = bool(cfg.get("training", {}).get("bf16", True))

    kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "attn_implementation": attn_impl,
        "device_map": "auto",
    }
    if qlora_mode:
        q_cfg = cfg["fallback"]["qlora"]
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=q_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=bool(q_cfg.get("bnb_4bit_use_double_quant", True)),
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        kwargs["torch_dtype"] = torch.bfloat16 if use_bf16 else torch.float32

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except (ImportError, RuntimeError, ValueError) as exc:
        msg = str(exc).lower()
        if kwargs.get("attn_implementation") == "flash_attention_2" and "flash" in msg:
            print(f"FlashAttention2 unavailable ({exc}); retrying with attn_implementation=sdpa")
            kwargs["attn_implementation"] = "sdpa"
            model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        else:
            raise
    if cfg["training"].get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
    return model


def _load_tokenizer(cfg: Dict[str, Any]) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _build_training_args(cfg: Dict[str, Any], output_dir: Path, per_device_train_batch_size: int):
    train_cfg = cfg["training"]

    common_kwargs = dict(
        output_dir=str(output_dir),
        logging_steps=int(train_cfg["logging_steps"]),
        save_steps=int(train_cfg["save_steps"]),
        eval_steps=int(train_cfg["eval_steps"]),
        save_total_limit=int(train_cfg.get("save_total_limit", 3)),
        learning_rate=float(train_cfg["learning_rate"]),
        lr_scheduler_type=str(train_cfg.get("lr_scheduler_type", "cosine")),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.05)),
        max_steps=int(train_cfg["max_steps"]),
        per_device_train_batch_size=int(per_device_train_batch_size),
        per_device_eval_batch_size=int(train_cfg.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        bf16=bool(train_cfg.get("bf16", True)),
        report_to=list(train_cfg.get("report_to", [])),
        logging_first_step=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        remove_unused_columns=False,
    )

    def _select_kwargs(params: Dict[str, inspect.Parameter]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in common_kwargs.items():
            if k in params:
                out[k] = v
        if "eval_strategy" in params and "evaluation_strategy" in common_kwargs:
            out["eval_strategy"] = common_kwargs["evaluation_strategy"]
        if "logging_strategy" in params and "logging_strategy" not in out:
            out["logging_strategy"] = "steps"
        return out

    try:
        from trl import SFTConfig
        sig = inspect.signature(SFTConfig.__init__).parameters
        kwargs = _select_kwargs(sig)
        if "dataset_text_field" in sig:
            kwargs["dataset_text_field"] = "text"
        if "max_seq_length" in sig:
            kwargs["max_seq_length"] = int(train_cfg["max_seq_length"])
        elif "max_length" in sig:
            kwargs["max_length"] = int(train_cfg["max_seq_length"])
        if "packing" in sig:
            kwargs["packing"] = bool(train_cfg.get("packing", False))

        return SFTConfig(**kwargs)
    except Exception:
        from transformers import TrainingArguments
        sig = inspect.signature(TrainingArguments.__init__).parameters
        kwargs = _select_kwargs(sig)
        return TrainingArguments(**kwargs)


def _prepare_datasets(tokenizer: Any, data_dir: Path):
    train_path = data_dir / "train.jsonl"
    dev_path = data_dir / "dev.jsonl"

    ds = load_dataset(
        "json",
        data_files={
            "train": str(train_path),
            "dev": str(dev_path),
        },
    )

    def add_text(batch):
        texts: List[str] = []
        for msgs in batch["messages"]:
            text = _render_chat(tokenizer, msgs, add_generation_prompt=False)
            texts.append(text)
        return {"text": texts}

    ds = ds.map(add_text, batched=True)
    return ds["train"], ds["dev"]


def _render_chat(tokenizer: Any, messages: List[Dict[str, str]], add_generation_prompt: bool) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            pass
    chunks: List[str] = []
    for turn in messages:
        chunks.append(f"{turn['role']}: {turn['content']}")
    if add_generation_prompt:
        chunks.append("assistant:")
    return "\n".join(chunks)


def _build_prompt_turns(messages: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], str]:
    if not messages:
        return [], ""
    if messages[-1].get("role") != "assistant":
        return [], ""
    answer = str(messages[-1].get("content", "")).strip()
    prompt_turns: List[Dict[str, str]] = []
    for turn in messages[:-1]:
        role = str(turn.get("role", "")).strip()
        content = str(turn.get("content", "")).strip()
        if role in {"system", "user"} and content:
            prompt_turns.append({"role": role, "content": content})
    if not prompt_turns:
        return [], ""
    return prompt_turns, answer


def _extract_first_present(row: Dict[str, Any], candidates: Sequence[str]) -> Optional[str]:
    for key in candidates:
        if key in row and row[key] is not None:
            text = str(row[key]).strip()
            if text:
                return text
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


def _parse_choice(text: str) -> Optional[str]:
    match = re.search(r"\b([A-E])\b", text.upper())
    if match:
        return match.group(1)
    return None


def _is_strict_mcq_output(text: str) -> bool:
    return bool(re.fullmatch(r"\s*[A-E](?:[.)])?\s*", text))


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


def _build_mcq_eval_subset(cfg: Dict[str, Any], seed: int, target_samples: int) -> List[Dict[str, Any]]:
    if target_samples <= 0:
        return []

    sel_cfg = cfg.get("selection", {})
    mcq_cfg = sel_cfg.get("mcq", {})
    if not bool(mcq_cfg.get("enabled", True)):
        return []

    dataset_id = str(mcq_cfg.get("dataset_id", "CohereLabs/Global-MMLU-Lite"))
    split = str(mcq_cfg.get("split", "dev"))
    assert_no_benchmark_test_split(dataset_id=dataset_id, split=split, purpose="SFT composite MCQ selection")
    print(f"[selection] MCQ source verified for training-time use: dataset_id={dataset_id}, split={split}")
    langs_raw = mcq_cfg.get("langs", ["ja", "ko"])
    langs = [normalize_lang(str(lang)) for lang in langs_raw]
    langs = [lang for lang in langs if lang]
    if not langs:
        return []

    lang_field_candidates = [str(x) for x in mcq_cfg.get("lang_field_candidates", ["language", "lang"])]
    id_fields = [str(x) for x in mcq_cfg.get("id_fields", ["id", "sample_id", "example_id", "uuid"])]
    fewshot_k = max(0, int(mcq_cfg.get("fewshot_k", 2)))
    max_choices = max(2, int(mcq_cfg.get("max_choices", 4)))
    max_rows_per_lang = max(32, int(mcq_cfg.get("max_rows_per_lang", 512)))
    per_lang_target = max(1, int(mcq_cfg.get("samples_per_lang", 0)) or (target_samples + len(langs) - 1) // len(langs))

    rng = random.Random(seed + 17)
    out: List[Dict[str, Any]] = []

    for lang in langs:
        try:
            rows = _load_language_rows(
                dataset_id=dataset_id,
                split=split,
                lang=lang,
                lang_field_candidates=lang_field_candidates,
                max_rows=max_rows_per_lang,
            )
        except Exception as exc:
            print(f"[selection] MCQ subset skipped for {lang}: {exc}")
            continue

        parsed: List[Dict[str, Any]] = []
        for row in rows:
            item = _extract_mcq(row, id_fields=id_fields)
            if not item:
                continue
            if len(item["choices"]) > max_choices:
                continue
            parsed.append(item)

        if not parsed:
            continue

        rng.shuffle(parsed)
        fewshot = parsed[:fewshot_k]
        selected = parsed[fewshot_k : fewshot_k + per_lang_target] if len(parsed) > fewshot_k else parsed[:per_lang_target]
        for sample in selected:
            prompt_turns = [{"role": "user", "content": _choice_prompt(sample["question"], sample["choices"], fewshot)}]
            out.append(
                {
                    "prompt_turns": prompt_turns,
                    "reference": sample["answer"],
                    "answer": sample["answer"],
                    "task": "mcq_format",
                    "lang": str(lang),
                    "sample_id": sample.get("sample_id"),
                }
            )

    rng.shuffle(out)
    return out[:target_samples]


def _build_composite_eval_subset(dev_ds: Any, cfg: Dict[str, Any], seed: int) -> Dict[str, List[Dict[str, Any]]]:
    sel_cfg = cfg.get("selection", {})
    sample_cfg = sel_cfg.get("samples", {})
    target = {
        "translation": int(sample_cfg.get("translation", 24)),
        "language_consistency": int(sample_cfg.get("language_consistency", 24)),
        "entity": int(sample_cfg.get("entity", 24)),
        "structured": int(sample_cfg.get("structured", 24)),
        "mcq_format": int(sample_cfg.get("mcq_format", 16)),
    }

    buckets: Dict[str, List[Dict[str, Any]]] = {k: [] for k in target}
    if len(dev_ds) == 0:
        return buckets

    indices = list(range(len(dev_ds)))
    random.Random(seed).shuffle(indices)

    for idx in indices:
        row = dev_ds[idx]
        messages = row.get("messages") or []
        if not isinstance(messages, list):
            continue
        prompt_turns, reference = _build_prompt_turns(messages)
        if not prompt_turns or not reference:
            continue

        task = str(row.get("task", "instruction"))
        lang = str(row.get("lang", "en"))
        base_item = {
            "prompt_turns": prompt_turns,
            "reference": reference,
            "task": task,
            "lang": lang,
        }

        if task == "translation" and len(buckets["translation"]) < target["translation"]:
            buckets["translation"].append(base_item)

        if lang in {"ja", "ko"} and len(buckets["language_consistency"]) < target["language_consistency"]:
            buckets["language_consistency"].append(base_item)

        if task == "entity" and len(buckets["entity"]) < target["entity"]:
            entity_key = str(row.get("entity_key", "")).strip()
            if not entity_key:
                entity_key = reference.split()[0] if reference.split() else reference[:8]
            item = dict(base_item)
            item["entity_key"] = entity_key
            buckets["entity"].append(item)

        if task == "structured" and len(buckets["structured"]) < target["structured"]:
            required_keys = row.get("required_keys")
            if not isinstance(required_keys, list) or not required_keys:
                required_keys = ["company", "revenue_2024_jpy", "revenue_2025_jpy", "yoy_growth_pct"]
            item = dict(base_item)
            item["required_keys"] = [str(k) for k in required_keys]
            buckets["structured"].append(item)

        if all(len(buckets[name]) >= target[name] for name in ["translation", "language_consistency", "entity", "structured"]):
            break

    if target["mcq_format"] > 0:
        buckets["mcq_format"] = _build_mcq_eval_subset(cfg, seed, target["mcq_format"])

    return buckets


def _generate_for_selection(
    model: Any,
    tokenizer: Any,
    prompt_turns: List[Dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    prompt = _render_chat(tokenizer, prompt_turns, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = temperature > 0
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

    gen_ids = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def _json_has_keys(text: str, required_keys: List[str]) -> bool:
    candidate = text
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        candidate = match.group(0)
    try:
        obj = json.loads(candidate)
    except Exception:
        return False
    return isinstance(obj, dict) and all(k in obj for k in required_keys)


def _score_composite(
    model: Any,
    tokenizer: Any,
    eval_subset: Dict[str, List[Dict[str, Any]]],
    cfg: Dict[str, Any],
) -> Tuple[float, Dict[str, float]]:
    from sacrebleu.metrics import CHRF

    sel_cfg = cfg.get("selection", {})
    w_cfg = sel_cfg.get("weights", {})
    weights = {
        "translation": float(w_cfg.get("translation", 0.40)),
        "language_consistency": float(w_cfg.get("language_consistency", 0.30)),
        "entity": float(w_cfg.get("entity", 0.20)),
        "structured": float(w_cfg.get("structured", 0.10)),
        "mcq_format": float(w_cfg.get("mcq_format", 0.0)),
    }
    gen_cfg = sel_cfg.get("generation", {})
    max_new_tokens = int(gen_cfg.get("max_new_tokens", 128))
    temperature = float(gen_cfg.get("temperature", 0.0))
    top_p = float(gen_cfg.get("top_p", 1.0))

    chrf = CHRF(word_order=2)

    translation_scores: List[float] = []
    for item in eval_subset.get("translation", []):
        pred = _generate_for_selection(model, tokenizer, item["prompt_turns"], max_new_tokens, temperature, top_p)
        score = chrf.sentence_score(pred, [item["reference"]]).score / 100.0
        translation_scores.append(max(0.0, min(1.0, float(score))))

    language_scores: List[float] = []
    for item in eval_subset.get("language_consistency", []):
        pred = _generate_for_selection(model, tokenizer, item["prompt_turns"], max_new_tokens, temperature, top_p)
        confused = language_confused(str(item["lang"]), pred)
        language_scores.append(0.0 if confused else 1.0)

    entity_scores: List[float] = []
    for item in eval_subset.get("entity", []):
        pred = _generate_for_selection(model, tokenizer, item["prompt_turns"], max_new_tokens, temperature, top_p)
        entity_scores.append(1.0 if str(item.get("entity_key", "")).strip() in pred else 0.0)

    structured_scores: List[float] = []
    for item in eval_subset.get("structured", []):
        pred = _generate_for_selection(model, tokenizer, item["prompt_turns"], max_new_tokens, temperature, top_p)
        structured_scores.append(1.0 if _json_has_keys(pred, item.get("required_keys", [])) else 0.0)

    mcq_scores: List[float] = []
    mcq_accuracy_scores: List[float] = []
    mcq_strict_scores: List[float] = []
    for item in eval_subset.get("mcq_format", []):
        pred = _generate_for_selection(model, tokenizer, item["prompt_turns"], max_new_tokens, temperature, top_p)
        pred_choice = _parse_choice(pred)
        strict_ok = _is_strict_mcq_output(pred)
        correct = pred_choice == str(item.get("answer", "")).strip().upper()
        # Balance strict format compliance and answer correctness so we don't select on benchmark-only behavior.
        mcq_scores.append((0.5 if strict_ok else 0.0) + (0.5 if correct else 0.0))
        mcq_accuracy_scores.append(1.0 if correct else 0.0)
        mcq_strict_scores.append(1.0 if strict_ok else 0.0)

    components = {
        "translation": float(sum(translation_scores) / max(len(translation_scores), 1)),
        "language_consistency": float(sum(language_scores) / max(len(language_scores), 1)),
        "entity": float(sum(entity_scores) / max(len(entity_scores), 1)),
        "structured": float(sum(structured_scores) / max(len(structured_scores), 1)),
        "mcq_format": float(sum(mcq_scores) / max(len(mcq_scores), 1)),
        "mcq_accuracy": float(sum(mcq_accuracy_scores) / max(len(mcq_accuracy_scores), 1)),
        "mcq_strict_format": float(sum(mcq_strict_scores) / max(len(mcq_strict_scores), 1)),
    }

    composite = (
        weights["translation"] * components["translation"]
        + weights["language_consistency"] * components["language_consistency"]
        + weights["entity"] * components["entity"]
        + weights["structured"] * components["structured"]
        + weights["mcq_format"] * components["mcq_format"]
    )

    details = {
        "composite": float(composite),
        "translation": components["translation"],
        "language_consistency": components["language_consistency"],
        "entity": components["entity"],
        "structured": components["structured"],
        "mcq_format": components["mcq_format"],
        "mcq_accuracy": components["mcq_accuracy"],
        "mcq_strict_format": components["mcq_strict_format"],
        "n_translation": float(len(translation_scores)),
        "n_language_consistency": float(len(language_scores)),
        "n_entity": float(len(entity_scores)),
        "n_structured": float(len(structured_scores)),
        "n_mcq_format": float(len(mcq_scores)),
    }
    return float(composite), details


class CompositeSelectionCallback(TrainerCallback):
    def __init__(
        self,
        run_dir: Path,
        eval_subset: Dict[str, List[Dict[str, Any]]],
        cfg: Dict[str, Any],
        tokenizer: Any,
    ) -> None:
        self.run_dir = run_dir
        self.eval_subset = eval_subset
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.best_score: Optional[float] = None
        self.best_step: Optional[int] = None
        self.best_metrics: Dict[str, float] = {}
        self.best_adapter_dir = run_dir / "adapter_composite_best"
        self.history_path = run_dir / "composite_selection_history.jsonl"

    def _append_history(self, payload: Dict[str, Any]) -> None:
        with open(self.history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            return control

        was_training = bool(model.training)
        model.eval()
        try:
            score, details = _score_composite(model, self.tokenizer, self.eval_subset, self.cfg)
            payload = {
                "step": int(state.global_step),
                **details,
            }
            self._append_history(payload)

            if self.best_score is None or score > self.best_score:
                self.best_score = score
                self.best_step = int(state.global_step)
                self.best_metrics = details
                if self.best_adapter_dir.exists():
                    shutil.rmtree(self.best_adapter_dir)
                model.save_pretrained(str(self.best_adapter_dir))
                self.tokenizer.save_pretrained(str(self.best_adapter_dir))
        except Exception as exc:
            self._append_history({"step": int(state.global_step), "error": str(exc)})
        finally:
            if was_training:
                model.train()

        return control


def _train_once(
    cfg: Dict[str, Any],
    run_dir: Path,
    data_dir: Path,
    qlora_mode: bool = False,
    batch_override: int | None = None,
    resume_from_checkpoint: str | None = None,
    init_adapter_dir: Path | None = None,
):
    from trl import SFTTrainer

    tokenizer = _load_tokenizer(cfg)
    train_ds, dev_ds = _prepare_datasets(tokenizer, data_dir)

    model = _load_model(cfg, qlora_mode=qlora_mode)
    if init_adapter_dir and init_adapter_dir.exists():
        model = PeftModel.from_pretrained(model, str(init_adapter_dir), is_trainable=True)
    else:
        lora_cfg = _build_lora_config(cfg, qlora_mode=qlora_mode)
        model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    per_device_train_batch_size = int(batch_override or cfg["training"]["per_device_train_batch_size"])
    args = _build_training_args(cfg, output_dir=run_dir / "checkpoints", per_device_train_batch_size=per_device_train_batch_size)

    selection_cfg = cfg.get("selection", {})
    use_composite = bool(selection_cfg.get("enabled", True))
    eval_subset = _build_composite_eval_subset(dev_ds, cfg, int(cfg.get("seed", 42))) if use_composite else {}
    has_subset = any(len(v) > 0 for v in eval_subset.values())

    train_ds_for_trainer = train_ds
    dev_ds_for_trainer = dev_ds
    if not getattr(tokenizer, "chat_template", None):
        keep_cols = {"text"}
        remove_train = [c for c in train_ds.column_names if c not in keep_cols]
        remove_dev = [c for c in dev_ds.column_names if c not in keep_cols]
        if remove_train:
            train_ds_for_trainer = train_ds.remove_columns(remove_train)
        if remove_dev:
            dev_ds_for_trainer = dev_ds.remove_columns(remove_dev)

    callback = CompositeSelectionCallback(run_dir, eval_subset, cfg, tokenizer) if (use_composite and has_subset) else None
    callbacks = [callback] if callback else None

    trainer_sig = inspect.signature(SFTTrainer.__init__).parameters
    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": train_ds_for_trainer,
        "eval_dataset": dev_ds_for_trainer,
    }
    if "tokenizer" in trainer_sig:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_sig:
        trainer_kwargs["processing_class"] = tokenizer
    if callbacks and "callbacks" in trainer_sig:
        trainer_kwargs["callbacks"] = callbacks

    trainer = SFTTrainer(**trainer_kwargs)

    if resume_from_checkpoint:
        resume_path = Path(resume_from_checkpoint)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
        trainer.train(resume_from_checkpoint=str(resume_path))
    else:
        trainer.train()

    adapter_dir = run_dir / "adapter"
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)

    selection_summary: Dict[str, Any] = {
        "method": "eval_loss",
        "composite_enabled": bool(use_composite),
        "composite_has_subset": bool(has_subset),
    }

    if callback and callback.best_score is None and has_subset:
        score, details = _score_composite(trainer.model, tokenizer, eval_subset, cfg)
        callback.best_score = score
        callback.best_step = int(trainer.state.global_step)
        callback.best_metrics = details
        if callback.best_adapter_dir.exists():
            shutil.rmtree(callback.best_adapter_dir)
        trainer.model.save_pretrained(str(callback.best_adapter_dir))
        tokenizer.save_pretrained(str(callback.best_adapter_dir))

    if callback and callback.best_score is not None and callback.best_adapter_dir.exists():
        shutil.copytree(callback.best_adapter_dir, adapter_dir)
        selection_summary.update(
            {
                "method": "weighted_composite",
                "best_composite_score": float(callback.best_score),
                "best_step": int(callback.best_step or 0),
                "best_components": callback.best_metrics,
            }
        )
    else:
        trainer.save_model(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))

    if not (adapter_dir / "tokenizer_config.json").exists():
        tokenizer.save_pretrained(str(adapter_dir))

    metrics = trainer.evaluate()
    metrics["selection_method"] = selection_summary["method"]
    if "best_composite_score" in selection_summary:
        metrics["best_composite_score"] = selection_summary["best_composite_score"]

    with open(run_dir / "sft_eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(run_dir / "sft_selection_summary.json", "w", encoding="utf-8") as f:
        json.dump(selection_summary, f, indent=2)

    return selection_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Tiny Aya JA/KO SFT adapter with TRL")
    parser.add_argument("--config", default="training/configs/tiny_aya_ja_ko_sft.yaml")
    parser.add_argument("--data-dir", required=True, help="Directory containing train.jsonl/dev.jsonl")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--force-qlora", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None, help="Checkpoint directory to resume from")
    parser.add_argument("--init-adapter-dir", default=None, help="Optional adapter directory to continue training from")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)

    run_id = args.run_id or timestamp_run_id(cfg.get("run_prefix", "tiny_aya_ja_ko_sft"))
    output_root = Path(args.output_root or cfg["paths"]["output_root"])
    run_dir = ensure_dir(output_root / run_id / "artifacts" / "sft")
    data_dir = Path(args.data_dir)
    resume_from_checkpoint = str(Path(args.resume_from_checkpoint).expanduser().resolve()) if args.resume_from_checkpoint else None
    init_adapter_dir = Path(args.init_adapter_dir).expanduser().resolve() if args.init_adapter_dir else None
    if args.init_adapter_dir and (not init_adapter_dir or not init_adapter_dir.exists()):
        raise FileNotFoundError(f"--init-adapter-dir does not exist: {args.init_adapter_dir}")

    dump_yaml(cfg, run_dir / "resolved_config.yaml")

    state = {
        "started_at_utc": now_utc_iso(),
        "run_id": run_id,
        "data_dir": str(data_dir),
        "resume_from_checkpoint": resume_from_checkpoint,
        "init_adapter_dir": str(init_adapter_dir) if init_adapter_dir else None,
        "qlora_fallback_triggered": False,
        "selection_method": "eval_loss",
    }

    try:
        if args.force_qlora:
            state["selection"] = _train_once(
                cfg,
                run_dir,
                data_dir,
                qlora_mode=True,
                resume_from_checkpoint=resume_from_checkpoint,
                init_adapter_dir=init_adapter_dir,
            )
            state["qlora_fallback_triggered"] = True
        else:
            state["selection"] = _train_once(
                cfg,
                run_dir,
                data_dir,
                qlora_mode=False,
                resume_from_checkpoint=resume_from_checkpoint,
                init_adapter_dir=init_adapter_dir,
            )
    except RuntimeError as exc:
        err_str = str(exc).lower()
        fallback_cfg = cfg.get("fallback", {}).get("qlora", {})
        should_fallback = bool(cfg.get("fallback", {}).get("enabled", True)) and "out of memory" in err_str

        with open(run_dir / "first_failure.log", "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())

        if not should_fallback:
            raise

        print("OOM detected during LoRA SFT. Retrying with QLoRA fallback.")
        state["qlora_fallback_triggered"] = True

        torch.cuda.empty_cache()
        batch_fallback = int(fallback_cfg.get("per_device_train_batch_size", 1))
        state["selection"] = _train_once(
            cfg,
            run_dir,
            data_dir,
            qlora_mode=True,
            batch_override=batch_fallback,
            init_adapter_dir=init_adapter_dir,
        )

    if isinstance(state.get("selection"), dict):
        state["selection_method"] = state["selection"].get("method", state["selection_method"])

    state["completed_at_utc"] = now_utc_iso()
    with open(run_dir / "run_state.json", "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    print(f"SFT run complete: {run_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
