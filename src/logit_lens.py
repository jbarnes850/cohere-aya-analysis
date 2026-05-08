"""Logit lens utilities for Aya-family causal LMs."""

import gc
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.prompts import LANGUAGE_NAMES, Prompt, get_all_prompts

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    model_id: str,
    device: str = "cuda",
    dtype: torch.dtype = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer."""
    if dtype is None:
        if device == "cpu":
            dtype = torch.float32
        elif device == "mps":
            dtype = torch.float16
        else:
            dtype = torch.bfloat16
    logger.info("Loading model: %s (dtype=%s, device=%s)", model_id, dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to("mps")
    else:
        device_map = "auto" if device != "cpu" else {"": "cpu"}
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
        )
    model.eval()
    return model, tokenizer


class LogitLensAnalyzer:
    """Logit lens analyzer."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.hidden_states: Dict[int, torch.Tensor] = {}
        self._hooks = []

        if hasattr(model, "model") and hasattr(model.model, "layers"):
            self.layers = model.model.layers
            self.n_layers = len(self.layers)
            self.norm = model.model.norm
            self.lm_head = model.lm_head
        else:
            raise ValueError(f"Unsupported model architecture: {type(model)}")

        self.logit_scale = getattr(model.config, "logit_scale", 1.0)

        logger.info(
            "LogitLensAnalyzer initialized: %d layers, logit_scale=%.4f",
            self.n_layers, self.logit_scale,
        )

    def _register_hooks(self):
        self._clear_hooks()
        self.hidden_states = {}

        for layer_idx in range(self.n_layers):
            hook = self.layers[layer_idx].register_forward_hook(
                self._make_hook(layer_idx)
            )
            self._hooks.append(hook)

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            self.hidden_states[layer_idx] = hidden[0, -1, :].detach()
        return hook_fn

    def _clear_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self.hidden_states = {}

    def project_to_vocab(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Project a hidden state to vocab logits."""
        normed = self.norm(hidden_state.unsqueeze(0))
        logits = self.lm_head(normed) * self.logit_scale
        return logits.squeeze(0)

    def _get_target_token_ids(self, target: str) -> List[int]:
        """Tokenize a target string."""
        return self.tokenizer.encode(target, add_special_tokens=False)

    def _match_target_in_topk(
        self,
        target: str,
        target_first_id: int,
        top_ids: List[int],
        top_tokens: List[str],
    ) -> bool:
        """Check if target appears in top-k predictions."""
        if target_first_id in top_ids:
            return True
        target_lower = target.lower()
        for tok in top_tokens:
            tok_lower = tok.lower().strip()
            if not tok_lower:
                continue
            if target_lower in tok_lower:
                return True
            if len(tok_lower) >= 1 and tok_lower in target_lower:
                return True
        return False

    @torch.no_grad()
    def analyze_prompt(
        self,
        prompt: Prompt,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """Run logit lens on a single prompt."""
        self._register_hooks()

        input_ids = self.tokenizer.encode(prompt.text, return_tensors="pt").to(self.device)
        _ = self.model(input_ids)

        layer_results = []
        for layer_idx in range(self.n_layers):
            hidden = self.hidden_states[layer_idx]
            logits = self.project_to_vocab(hidden)
            probs = torch.softmax(logits.float(), dim=-1)

            top_probs, top_ids = torch.topk(probs, top_k)
            top_tokens = [self.tokenizer.decode([tid.item()]).strip() for tid in top_ids]

            layer_results.append({
                "layer": layer_idx,
                "top_tokens": top_tokens,
                "top_probs": top_probs.cpu().numpy().tolist(),
                "top_ids": top_ids.cpu().numpy().tolist(),
            })

        self._clear_hooks()

        emergence_layer = None
        crystallization_layer = None
        emergence_evidence = None
        crystallization_evidence = None
        target_token_ids = None
        target_n_tokens = None

        if prompt.expected_next:
            target = prompt.expected_next
            target_token_ids = self._get_target_token_ids(target)
            target_n_tokens = len(target_token_ids)
            target_first_id = target_token_ids[0]

            for lr in layer_results:
                top_ids_list = lr["top_ids"]
                top_tokens_list = lr["top_tokens"]

                if emergence_layer is None:
                    if self._match_target_in_topk(
                        target, target_first_id, top_ids_list, top_tokens_list
                    ):
                        emergence_layer = lr["layer"]
                        emergence_evidence = (
                            f"top10=[{', '.join(top_tokens_list)}]"
                        )

                if crystallization_layer is None:
                    if top_ids_list[0] == target_first_id:
                        crystallization_layer = lr["layer"]
                        crystallization_evidence = (
                            f"top1={top_tokens_list[0]}"
                        )
                    elif crystallization_layer is None:
                        tok_lower = top_tokens_list[0].lower().strip()
                        target_lower = target.lower()
                        if tok_lower and (
                            target_lower in tok_lower or tok_lower in target_lower
                        ):
                            crystallization_layer = lr["layer"]
                            crystallization_evidence = (
                                f"top1={top_tokens_list[0]} (string match)"
                            )

        language_gap = None
        if emergence_layer is not None and crystallization_layer is not None:
            language_gap = crystallization_layer - emergence_layer

        return {
            "prompt": prompt.text,
            "language": prompt.language,
            "category": prompt.category,
            "expected_next": prompt.expected_next,
            "layer_results": layer_results,
            "emergence_layer": emergence_layer,
            "crystallization_layer": crystallization_layer,
            "language_gap": language_gap,
            "emergence_evidence": emergence_evidence,
            "crystallization_evidence": crystallization_evidence,
            "target_n_tokens": target_n_tokens,
            "n_layers": self.n_layers,
            "final_prediction": layer_results[-1]["top_tokens"][0] if layer_results else None,
        }

    def analyze_all_prompts(
        self,
        prompts: Optional[List[Prompt]] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Run logit lens on all prompts."""
        if prompts is None:
            prompts = get_all_prompts()

        results = []
        for i, prompt in enumerate(prompts):
            logger.info("Analyzing prompt %d/%d: [%s] %s...",
                        i + 1, len(prompts), prompt.language, prompt.text[:40])
            result = self.analyze_prompt(prompt, top_k=top_k)
            results.append(result)

        return results


def build_logit_lens_summary(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Build summary DataFrame."""
    rows = []
    for r in results:
        rows.append({
            "language": r["language"],
            "language_name": LANGUAGE_NAMES.get(r["language"], r["language"]),
            "category": r["category"],
            "emergence_layer": r["emergence_layer"],
            "crystallization_layer": r["crystallization_layer"],
            "language_gap": r["language_gap"],
            "final_prediction": r["final_prediction"],
            "expected_next": r["expected_next"],
            "emergence_evidence": r.get("emergence_evidence", ""),
            "crystallization_evidence": r.get("crystallization_evidence", ""),
            "target_n_tokens": r.get("target_n_tokens", None),
            "n_layers": r["n_layers"],
        })
    return pd.DataFrame(rows)


def verify_target_tokenization(
    tokenizer: AutoTokenizer,
    prompts: Optional[List[Prompt]] = None,
) -> pd.DataFrame:
    """Verify how each expected_next tokenizes."""
    if prompts is None:
        prompts = get_all_prompts()

    rows = []
    for p in prompts:
        if not p.expected_next:
            continue
        ids = tokenizer.encode(p.expected_next, add_special_tokens=False)
        decoded = [tokenizer.decode([i]).strip() for i in ids]
        rows.append({
            "language": p.language,
            "category": p.category,
            "target": p.expected_next,
            "n_tokens": len(ids),
            "token_ids": str(ids),
            "decoded_parts": str(decoded),
            "first_token": decoded[0] if decoded else "",
            "single_token": len(ids) == 1,
        })

    df = pd.DataFrame(rows)
    n_multi = (~df["single_token"]).sum()
    n_total = len(df)
    logger.info(
        "Target tokenization: %d/%d single-token, %d/%d multi-token",
        n_total - n_multi, n_total, n_multi, n_total,
    )
    return df


def build_layer_token_matrix(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Build a matrix of top-1 predicted tokens by layer."""
    rows = []
    for r in results:
        for lr in r["layer_results"]:
            rows.append({
                "language": r["language"],
                "category": r["category"],
                "layer": lr["layer"],
                "top1_token": lr["top_tokens"][0],
                "top1_prob": lr["top_probs"][0],
                "prompt_text": r["prompt"][:50],
            })
    return pd.DataFrame(rows)


def compute_multilingual_entropy(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Compute per-layer entropy across languages."""
    n_layers = results[0]["n_layers"] if results else 0
    rows = []

    for layer_idx in range(n_layers):
        predictions_by_lang = {}
        for r in results:
            lang = r["language"]
            if layer_idx < len(r["layer_results"]):
                lr = r["layer_results"][layer_idx]
                predictions_by_lang.setdefault(lang, []).append(lr["top_tokens"][0])

        all_tokens = []
        for lang_tokens in predictions_by_lang.values():
            all_tokens.extend(lang_tokens)

        unique_tokens = set(all_tokens)
        n_unique = len(unique_tokens)
        n_total = len(all_tokens)

        if n_total > 0:
            token_counts = {}
            for t in all_tokens:
                token_counts[t] = token_counts.get(t, 0) + 1
            probs = np.array(list(token_counts.values())) / n_total
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
        else:
            entropy = 0.0

        rows.append({
            "layer": layer_idx,
            "n_unique_tokens": n_unique,
            "n_total_predictions": n_total,
            "entropy": entropy,
        })

    return pd.DataFrame(rows)


def run_logit_lens_with_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    output_dir: str = "outputs",
    device: str = "cuda",
    top_k: int = 10,
) -> Dict[str, Any]:
    """Run logit lens analysis using a pre-loaded model."""
    output_path = Path(output_dir)
    (output_path / "tables").mkdir(parents=True, exist_ok=True)

    analyzer = LogitLensAnalyzer(model, tokenizer, device=device)

    prompts = get_all_prompts()
    logger.info("Running logit lens on %d prompts across %d layers", len(prompts), analyzer.n_layers)

    target_tok_df = verify_target_tokenization(tokenizer, prompts)
    target_tok_df.to_csv(output_path / "tables" / "target_tokenization.csv", index=False)
    n_multi = (~target_tok_df["single_token"]).sum()
    if n_multi > 0:
        logger.warning(
            "%d/%d targets are multi-token. Detection uses first-token ID matching.",
            n_multi, len(target_tok_df),
        )
        print("\n=== Multi-Token Targets (first-token matching will be used) ===")
        print(target_tok_df[~target_tok_df["single_token"]].to_string(index=False))
        print()

    raw_results = analyzer.analyze_all_prompts(prompts, top_k=top_k)

    summary_df = build_logit_lens_summary(raw_results)
    layer_matrix_df = build_layer_token_matrix(raw_results)
    entropy_df = compute_multilingual_entropy(raw_results)

    summary_df.to_csv(output_path / "tables" / "logit_lens_summary.csv", index=False)
    layer_matrix_df.to_csv(output_path / "tables" / "logit_lens_layers.csv", index=False)
    entropy_df.to_csv(output_path / "tables" / "multilingual_entropy.csv", index=False)
    logger.info("Saved target_tokenization.csv, logit_lens_summary.csv, logit_lens_layers.csv, multilingual_entropy.csv")

    print("\n=== Logit Lens Summary ===")
    agg = summary_df.groupby("language_name").agg(
        mean_emergence=("emergence_layer", "mean"),
        mean_crystallization=("crystallization_layer", "mean"),
        mean_gap=("language_gap", "mean"),
    ).reset_index()
    print(agg.to_string(index=False))

    return {
        "summary": summary_df,
        "layer_matrix": layer_matrix_df,
        "entropy": entropy_df,
        "raw_results": raw_results,
    }


def run_logit_lens(
    model_id: str,
    output_dir: str = "outputs",
    device: str = "cuda",
    top_k: int = 10,
) -> Dict[str, Any]:
    """Run logit lens analysis (standalone)."""
    model, tokenizer = load_model_and_tokenizer(model_id, device=device)

    results = run_logit_lens_with_model(
        model, tokenizer, output_dir=output_dir, device=device, top_k=top_k,
    )

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


