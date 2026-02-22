#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.frontier_utils import dump_yaml, ensure_dir, load_yaml, now_utc_iso, set_global_seed, timestamp_run_id


def _load_model(cfg: Dict[str, Any], sft_adapter_dir: Path | None):
    use_bf16 = bool(cfg.get("training", {}).get("bf16", True))
    attn_impl = str(cfg["model"].get("attn_implementation", "flash_attention_2"))
    if attn_impl == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
        print("flash_attn not installed; falling back to attn_implementation=sdpa")
        attn_impl = "sdpa"
    model_kwargs = dict(
        torch_dtype=(torch.bfloat16 if use_bf16 else torch.float32),
        trust_remote_code=True,
        device_map="auto",
        attn_implementation=attn_impl,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(cfg["model"]["base_model"], **model_kwargs)
    except (ImportError, RuntimeError, ValueError) as exc:
        msg = str(exc).lower()
        if model_kwargs.get("attn_implementation") == "flash_attention_2" and "flash" in msg:
            print(f"FlashAttention2 unavailable ({exc}); retrying with attn_implementation=sdpa")
            model_kwargs["attn_implementation"] = "sdpa"
            model = AutoModelForCausalLM.from_pretrained(cfg["model"]["base_model"], **model_kwargs)
        else:
            raise

    if sft_adapter_dir and sft_adapter_dir.exists():
        model = PeftModel.from_pretrained(model, str(sft_adapter_dir), is_trainable=True)
        return model

    lora_cfg = cfg["lora"]
    peft_cfg = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["lora_alpha"]),
        lora_dropout=float(lora_cfg["lora_dropout"]),
        bias=str(lora_cfg.get("bias", "none")),
        task_type="CAUSAL_LM",
        target_modules=list(lora_cfg["target_modules"]),
        layers_to_transform=list(lora_cfg["layers_to_transform"]),
        modules_to_save=list(lora_cfg.get("modules_to_save", [])),
        layers_pattern=lora_cfg.get("layers_pattern", "layers"),
    )
    model = get_peft_model(model, peft_cfg)
    return model


def _training_args(cfg: Dict[str, Any], out_dir: Path):
    tcfg = cfg["training"]
    kwargs = dict(
        output_dir=str(out_dir),
        learning_rate=float(tcfg["learning_rate"]),
        lr_scheduler_type=str(tcfg.get("lr_scheduler_type", "cosine")),
        warmup_ratio=float(tcfg.get("warmup_ratio", 0.05)),
        max_steps=int(tcfg["max_steps"]),
        per_device_train_batch_size=int(tcfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(tcfg.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(tcfg["gradient_accumulation_steps"]),
        bf16=bool(tcfg.get("bf16", True)),
        evaluation_strategy="steps",
        eval_steps=int(tcfg.get("eval_steps", 50)),
        save_steps=int(tcfg.get("save_steps", 50)),
        logging_steps=int(tcfg.get("logging_steps", 10)),
        report_to=list(tcfg.get("report_to", [])),
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    def _select_kwargs(params: Dict[str, inspect.Parameter]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in kwargs.items():
            if k in params:
                out[k] = v
        if "eval_strategy" in params and "evaluation_strategy" in kwargs:
            out["eval_strategy"] = kwargs["evaluation_strategy"]
        if "logging_strategy" in params and "logging_strategy" not in out:
            out["logging_strategy"] = "steps"
        return out

    try:
        from trl import DPOConfig
        sig = inspect.signature(DPOConfig.__init__).parameters
        dpo_kwargs = _select_kwargs(sig)
        if "beta" in sig:
            dpo_kwargs["beta"] = float(cfg["dpo"].get("beta", 0.1))
        if "max_length" in sig:
            dpo_kwargs["max_length"] = int(tcfg.get("max_length", 3072))
        if "max_prompt_length" in sig:
            dpo_kwargs["max_prompt_length"] = int(tcfg.get("max_prompt_length", 2048))
        if "loss_type" in sig:
            dpo_kwargs["loss_type"] = str(cfg["dpo"].get("loss_type", "sigmoid"))

        return DPOConfig(**dpo_kwargs)
    except Exception:
        from transformers import TrainingArguments
        sig = inspect.signature(TrainingArguments.__init__).parameters
        ta_kwargs = _select_kwargs(sig)
        return TrainingArguments(**ta_kwargs)


def _build_trainer(model: Any, args: Any, train_ds: Any, eval_ds: Any, tokenizer: Any, beta: float):
    from trl import DPOTrainer

    sig = inspect.signature(DPOTrainer.__init__)
    trainer_kwargs = {
        "model": model,
        "ref_model": None,
        "args": args,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
    }
    if "beta" in sig.parameters:
        trainer_kwargs["beta"] = beta

    if "tokenizer" in sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer

    return DPOTrainer(**trainer_kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Tiny Aya JA/KO DPO stage")
    parser.add_argument("--config", default="training/configs/tiny_aya_ja_ko_dpo.yaml")
    parser.add_argument("--pref-data-dir", required=True, help="Directory containing train_pref.jsonl[/dev_pref.jsonl]")
    parser.add_argument("--sft-adapter-dir", required=False, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--remaining-minutes", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    set_global_seed(int(cfg.get("seed", 42)))
    started_at_utc = now_utc_iso()

    min_required = float(cfg.get("time_gate", {}).get("min_required_minutes", 90))
    if args.remaining_minutes is not None and args.remaining_minutes < min_required:
        print(
            f"Skipping DPO: remaining minutes {args.remaining_minutes:.1f} < required {min_required:.1f}."
        )
        return

    run_id = args.run_id or timestamp_run_id(cfg.get("run_prefix", "tiny_aya_ja_ko_dpo"))
    run_dir = ensure_dir(Path(args.output_root or cfg["paths"]["output_root"]) / run_id / "artifacts" / "dpo")
    pref_dir = Path(args.pref_data_dir)
    sft_adapter = Path(args.sft_adapter_dir) if args.sft_adapter_dir else None

    dump_yaml(cfg, run_dir / "resolved_config.yaml")

    train_file = pref_dir / "train_pref.jsonl"
    dev_file = pref_dir / "dev_pref.jsonl"

    data_files = {"train": str(train_file)}
    if dev_file.exists():
        data_files["dev"] = str(dev_file)

    ds = load_dataset("json", data_files=data_files)
    if "dev" not in ds:
        ds = ds["train"].train_test_split(test_size=0.02, seed=int(cfg.get("seed", 42)))
        train_ds = ds["train"]
        eval_ds = ds["test"]
    else:
        train_ds = ds["train"]
        eval_ds = ds["dev"]

    if len(eval_ds) == 0:
        split = train_ds.train_test_split(test_size=0.02, seed=int(cfg.get("seed", 42)))
        train_ds = split["train"]
        eval_ds = split["test"]

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _load_model(cfg, sft_adapter)
    model.print_trainable_parameters()

    train_args = _training_args(cfg, run_dir / "checkpoints")
    trainer = _build_trainer(
        model=model,
        args=train_args,
        train_ds=train_ds,
        eval_ds=eval_ds,
        tokenizer=tokenizer,
        beta=float(cfg["dpo"].get("beta", 0.1)),
    )

    trainer.train()
    trainer.save_model(str(run_dir / "adapter"))
    tokenizer.save_pretrained(str(run_dir / "adapter"))

    metrics = trainer.evaluate()
    with open(run_dir / "dpo_eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(run_dir / "run_state.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "started_at_utc": started_at_utc,
                "completed_at_utc": now_utc_iso(),
                "run_id": run_id,
                "pref_dir": str(pref_dir),
                "sft_adapter_dir": str(sft_adapter) if sft_adapter else None,
            },
            f,
            indent=2,
        )

    print(f"DPO run complete: {run_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_float32_matmul_precision("high")
    main()
