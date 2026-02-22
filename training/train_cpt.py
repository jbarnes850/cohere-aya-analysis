#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.frontier_utils import dump_yaml, ensure_dir, load_yaml, now_utc_iso, set_global_seed, timestamp_run_id


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
        save_total_limit=int(train_cfg.get("save_total_limit", 4)),
        learning_rate=float(train_cfg["learning_rate"]),
        lr_scheduler_type=str(train_cfg.get("lr_scheduler_type", "cosine")),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.02)),
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
        for key, value in common_kwargs.items():
            if key in params:
                out[key] = value
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
            kwargs["packing"] = bool(train_cfg.get("packing", True))
        return SFTConfig(**kwargs)
    except Exception:
        from transformers import TrainingArguments

        sig = inspect.signature(TrainingArguments.__init__).parameters
        kwargs = _select_kwargs(sig)
        return TrainingArguments(**kwargs)


def _prepare_datasets(data_dir: Path):
    train_path = data_dir / "train_text.jsonl"
    dev_path = data_dir / "dev_text.jsonl"
    if not train_path.exists() or not dev_path.exists():
        raise FileNotFoundError(f"CPT text files not found in {data_dir}")
    ds = load_dataset("json", data_files={"train": str(train_path), "dev": str(dev_path)})
    return ds["train"], ds["dev"]


def _train_once(
    cfg: Dict[str, Any],
    run_dir: Path,
    data_dir: Path,
    qlora_mode: bool = False,
    batch_override: Optional[int] = None,
    resume_from_checkpoint: Optional[str] = None,
    init_adapter_dir: Optional[Path] = None,
):
    from trl import SFTTrainer

    tokenizer = _load_tokenizer(cfg)
    train_ds, dev_ds = _prepare_datasets(data_dir)
    model = _load_model(cfg, qlora_mode=qlora_mode)

    if init_adapter_dir and init_adapter_dir.exists():
        model = PeftModel.from_pretrained(model, str(init_adapter_dir), is_trainable=True)
    else:
        model = get_peft_model(model, _build_lora_config(cfg, qlora_mode=qlora_mode))
    model.print_trainable_parameters()

    per_device_train_batch_size = int(batch_override or cfg["training"]["per_device_train_batch_size"])
    args = _build_training_args(cfg, output_dir=run_dir / "checkpoints", per_device_train_batch_size=per_device_train_batch_size)

    trainer_sig = inspect.signature(SFTTrainer.__init__).parameters
    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": train_ds,
        "eval_dataset": dev_ds,
    }
    if "tokenizer" in trainer_sig:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_sig:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)
    if resume_from_checkpoint:
        resume_path = Path(resume_from_checkpoint)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
        trainer.train(resume_from_checkpoint=str(resume_path))
    else:
        trainer.train()

    adapter_dir = run_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    metrics = trainer.evaluate()
    with open(run_dir / "cpt_eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Tiny Aya JA/KO continued pretraining adapter with TRL")
    parser.add_argument("--config", default="training/configs/tiny_aya_ja_ko_cpt.yaml")
    parser.add_argument("--data-dir", required=True, help="Directory containing train_text.jsonl/dev_text.jsonl")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--force-qlora", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--init-adapter-dir", default=None, help="Optional adapter directory to continue training from")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)

    run_id = args.run_id or timestamp_run_id(cfg.get("run_prefix", "tiny_aya_ja_ko_cpt"))
    output_root = Path(args.output_root or cfg["paths"]["output_root"])
    run_dir = ensure_dir(output_root / run_id / "artifacts" / "cpt")
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
    }

    try:
        if args.force_qlora:
            _train_once(
                cfg,
                run_dir,
                data_dir,
                qlora_mode=True,
                resume_from_checkpoint=resume_from_checkpoint,
                init_adapter_dir=init_adapter_dir,
            )
            state["qlora_fallback_triggered"] = True
        else:
            _train_once(
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
        print("OOM detected during LoRA CPT. Retrying with QLoRA fallback.")
        state["qlora_fallback_triggered"] = True
        torch.cuda.empty_cache()
        batch_fallback = int(fallback_cfg.get("per_device_train_batch_size", 1))
        _train_once(
            cfg,
            run_dir,
            data_dir,
            qlora_mode=True,
            batch_override=batch_fallback,
            init_adapter_dir=init_adapter_dir,
        )

    state["completed_at_utc"] = now_utc_iso()
    with open(run_dir / "run_state.json", "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    print(f"CPT run complete: {run_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
