"""CKA representation similarity on Aya Expanse 8B."""

import gc
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.prompts import ALL_CATEGORIES, Prompt

logger = logging.getLogger(__name__)

MODEL_ID = "CohereLabs/aya-expanse-8b"

LANGUAGE_PAIRS = [
    ("en", "en", "EN-EN"),   # Same-language upper bound (cross-category)
    ("ja", "ja", "JA-JA"),   # Same-language upper bound for CJK
    ("ja", "en", "JA-EN"),   # Primary interest
    ("ko", "en", "KO-EN"),   # Primary interest
    ("zh", "en", "ZH-EN"),   # CJK baseline
    ("ja", "zh", "JA-ZH"),   # CJK internal
    ("ja", "ko", "JA-KO"),   # CJK internal
    ("en", "vi", "EN-VI"),   # Latin script control
    ("en", "id", "EN-ID"),   # Latin script control
]

CKA_CATEGORIES = list(ALL_CATEGORIES.keys())


class CKAAnalyzer:
    """Compute linear CKA between language pairs."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.hidden_states: Dict[int, torch.Tensor] = {}
        self._hooks = []

        if hasattr(model, "model") and hasattr(model.model, "layers"):
            self.layers = model.model.layers
            self.n_layers = len(self.layers)
        else:
            raise ValueError(f"Unsupported model architecture: {type(model)}")

        logger.info("CKAAnalyzer initialized: %d layers", self.n_layers)

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

    @staticmethod
    def _debiased_hsic(K: torch.Tensor, L: torch.Tensor) -> float:
        """Compute debiased HSIC estimator."""
        n = K.shape[0]

        K_tilde = K.clone()
        L_tilde = L.clone()
        K_tilde.fill_diagonal_(0)
        L_tilde.fill_diagonal_(0)

        ones = torch.ones(n, 1, device=K.device)

        term1 = torch.trace(K_tilde @ L_tilde)
        term2 = (ones.T @ K_tilde @ ones @ ones.T @ L_tilde @ ones) / (
            (n - 1) * (n - 2)
        )
        term3 = (2.0 / (n - 2)) * ones.T @ K_tilde @ L_tilde @ ones

        return ((term1 + term2.item() - term3.item()) / (n * (n - 3))).item()

    @staticmethod
    def _compute_linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
        """Compute debiased linear CKA."""
        X = X.float()
        Y = Y.float()

        K = X @ X.T
        L = Y @ Y.T

        hsic_kl = CKAAnalyzer._debiased_hsic(K, L)
        hsic_kk = CKAAnalyzer._debiased_hsic(K, K)
        hsic_ll = CKAAnalyzer._debiased_hsic(L, L)

        hsic_kk = max(hsic_kk, 0.0)
        hsic_ll = max(hsic_ll, 0.0)

        denom = (hsic_kk * hsic_ll) ** 0.5
        if denom < 1e-10:
            return 0.0

        return hsic_kl / denom

    @torch.no_grad()
    def collect_hidden_states(
        self,
        prompts: List[Prompt],
    ) -> Dict[int, torch.Tensor]:
        """Collect last-position hidden states per layer."""
        self._register_hooks()

        all_states = {i: [] for i in range(self.n_layers)}

        for prompt in prompts:
            input_ids = self.tokenizer.encode(
                prompt.text, return_tensors="pt"
            ).to(self.device)
            _ = self.model(input_ids)

            for layer_idx in range(self.n_layers):
                all_states[layer_idx].append(
                    self.hidden_states[layer_idx].cpu()
                )

        self._clear_hooks()

        return {
            layer_idx: torch.stack(states)
            for layer_idx, states in all_states.items()
        }

    def _get_paired_prompts(
        self,
        lang_a: str,
        lang_b: str,
        same_language: bool = False,
    ) -> Tuple[List[Prompt], List[Prompt]]:
        """Get paired prompts across categories."""
        prompts_a = []
        prompts_b = []

        if same_language:
            half = len(CKA_CATEGORIES) // 2
            cats_a = CKA_CATEGORIES[:half]
            cats_b = CKA_CATEGORIES[half:half * 2]
            n_pairs = min(len(cats_a), len(cats_b))

            for i in range(n_pairs):
                cat_prompts_a = ALL_CATEGORIES[cats_a[i]]
                cat_prompts_b = ALL_CATEGORIES[cats_b[i]]
                if lang_a in cat_prompts_a and lang_b in cat_prompts_b:
                    prompts_a.append(cat_prompts_a[lang_a])
                    prompts_b.append(cat_prompts_b[lang_b])
        else:
            for cat_name in CKA_CATEGORIES:
                cat_prompts = ALL_CATEGORIES[cat_name]
                if lang_a in cat_prompts and lang_b in cat_prompts:
                    prompts_a.append(cat_prompts[lang_a])
                    prompts_b.append(cat_prompts[lang_b])

        return prompts_a, prompts_b

    def _bootstrap_cka(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        n_bootstrap: int = 1000,
    ) -> Tuple[float, float, float]:
        """Compute CKA with bootstrap confidence interval."""
        n = X.shape[0]
        cka_values = []

        for _ in range(n_bootstrap):
            indices = torch.randint(0, n, (n,))
            X_boot = X[indices]
            Y_boot = Y[indices]
            cka_values.append(self._compute_linear_cka(X_boot, Y_boot))

        cka_arr = np.array(cka_values)
        mean_cka = float(np.mean(cka_arr))
        ci_lower = float(np.percentile(cka_arr, 2.5))
        ci_upper = float(np.percentile(cka_arr, 97.5))

        return mean_cka, ci_lower, ci_upper

    def compute_pairwise_cka(
        self,
        n_bootstrap: int = 1000,
    ) -> Dict[str, Any]:
        """Compute CKA for all language pairs at each layer."""
        pair_similarities = {}
        pair_cis = {}
        n_prompts_per_pair = {}

        language_states: Dict[str, Dict[int, torch.Tensor]] = {}

        _cache_validation: Dict[str, List] = {}
        for lang_a, lang_b, label in LANGUAGE_PAIRS:
            if lang_a == lang_b:
                continue  # Same-language pairs are never cached
            prompts_a, prompts_b = self._get_paired_prompts(
                lang_a, lang_b, same_language=False
            )
            for lang, prompts in [(lang_a, prompts_a), (lang_b, prompts_b)]:
                key = f"{lang}_all"
                texts = [p.text for p in prompts]
                if key in _cache_validation:
                    assert _cache_validation[key] == texts, (
                        f"Caching invariant violated for {key}: pair {label} "
                        f"produces different prompts than a prior pair. "
                        f"Expected {len(_cache_validation[key])} prompts, "
                        f"got {len(texts)}."
                    )
                else:
                    _cache_validation[key] = texts

        for lang_a, lang_b, label in LANGUAGE_PAIRS:
            same_language = (lang_a == lang_b)
            prompts_a, prompts_b = self._get_paired_prompts(
                lang_a, lang_b, same_language=same_language
            )

            if len(prompts_a) < 2:
                logger.warning(
                    "Skipping %s: only %d paired prompts", label, len(prompts_a)
                )
                continue

            logger.info(
                "CKA %s: %d paired prompts, %s",
                label, len(prompts_a),
                "same-language (cross-category)" if same_language else "cross-language",
            )

            cache_key_a = f"{lang_a}_{'A' if same_language else 'all'}"
            cache_key_b = f"{lang_b}_{'B' if same_language else 'all'}"

            if same_language:
                states_a = self.collect_hidden_states(prompts_a)
                states_b = self.collect_hidden_states(prompts_b)
            else:
                if cache_key_a not in language_states:
                    language_states[cache_key_a] = self.collect_hidden_states(prompts_a)
                if cache_key_b not in language_states:
                    language_states[cache_key_b] = self.collect_hidden_states(prompts_b)
                states_a = language_states[cache_key_a]
                states_b = language_states[cache_key_b]

            n_prompts_per_pair[label] = len(prompts_a)

            layer_ckas = []
            layer_cis = []

            for layer_idx in range(self.n_layers):
                X = states_a[layer_idx]
                Y = states_b[layer_idx]
                mean_cka, ci_lower, ci_upper = self._bootstrap_cka(
                    X, Y, n_bootstrap=n_bootstrap
                )
                layer_ckas.append(mean_cka)
                layer_cis.append((ci_lower, ci_upper))

            pair_similarities[label] = layer_ckas
            pair_cis[label] = layer_cis

            logger.info(
                "  %s: mean CKA=%.3f, peak=%.3f (layer %d)",
                label,
                np.mean(layer_ckas),
                max(layer_ckas),
                int(np.argmax(layer_ckas)),
            )

        self._run_sanity_checks(language_states)

        return {
            "pair_similarities": pair_similarities,
            "pair_cis": pair_cis,
            "n_layers": self.n_layers,
            "n_prompts_per_pair": n_prompts_per_pair,
        }

    def _run_sanity_checks(
        self,
        language_states: Dict[str, Dict[int, torch.Tensor]],
    ):
        """Run basic sanity checks."""
        for key, states in language_states.items():
            mid_layer = self.n_layers // 2
            X = states[mid_layer]
            self_cka = self._compute_linear_cka(X, X)
            logger.info(
                "Sanity check: CKA(%s, %s) at layer %d = %.6f (expected 1.0)",
                key, key, mid_layer, self_cka,
            )
            if abs(self_cka - 1.0) > 1e-4:
                logger.warning(
                    "CKA self-similarity deviates from 1.0: %.6f", self_cka
                )
            break

        for key, states in language_states.items():
            mid_layer = self.n_layers // 2
            X = states[mid_layer]
            random_Y = torch.randn_like(X)
            random_cka = self._compute_linear_cka(X, random_Y)
            logger.info(
                "Sanity check: CKA(%s, random) at layer %d = %.6f (expected ~0.0)",
                key, mid_layer, random_cka,
            )
            if random_cka > 0.1:
                logger.warning(
                    "CKA with random is unexpectedly high: %.6f", random_cka
                )
            break


def run_representation_similarity(
    output_dir: str = "outputs",
    device: str = "cuda",
    model_id: str = MODEL_ID,
    model: Optional[AutoModelForCausalLM] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    n_bootstrap: int = 1000,
) -> Dict[str, Any]:
    """Run CKA representation similarity analysis."""
    output_path = Path(output_dir)
    (output_path / "tables").mkdir(parents=True, exist_ok=True)

    owns_model = model is None
    if owns_model:
        from src.logit_lens import load_model_and_tokenizer
        model, tokenizer = load_model_and_tokenizer(model_id, device=device)

    analyzer = CKAAnalyzer(model, tokenizer, device=device)
    logger.info(
        "Running CKA on %d language pairs across %d layers",
        len(LANGUAGE_PAIRS), analyzer.n_layers,
    )

    results = analyzer.compute_pairwise_cka(n_bootstrap=n_bootstrap)

    per_layer_rows = []
    for label, ckas in results["pair_similarities"].items():
        cis = results["pair_cis"][label]
        for layer_idx, (cka_val, (ci_lo, ci_hi)) in enumerate(zip(ckas, cis)):
            per_layer_rows.append({
                "pair": label,
                "layer": layer_idx,
                "cka_similarity": round(cka_val, 6),
                "ci_lower": round(ci_lo, 6),
                "ci_upper": round(ci_hi, 6),
            })

    per_layer_df = pd.DataFrame(per_layer_rows)
    per_layer_df.to_csv(
        output_path / "tables" / "representation_similarity.csv", index=False
    )

    summary_rows = []
    for label, ckas in results["pair_similarities"].items():
        cka_arr = np.array(ckas)
        summary_rows.append({
            "pair": label,
            "mean_cka": round(float(np.mean(cka_arr)), 4),
            "peak_layer": int(np.argmax(cka_arr)),
            "peak_cka": round(float(np.max(cka_arr)), 4),
            "trough_layer": int(np.argmin(cka_arr)),
            "trough_cka": round(float(np.min(cka_arr)), 4),
            "n_layers": len(ckas),
            "n_prompts": results["n_prompts_per_pair"].get(label, 0),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        output_path / "tables" / "representation_similarity_summary.csv",
        index=False,
    )

    logger.info(
        "Saved representation_similarity.csv (%d rows) and "
        "representation_similarity_summary.csv (%d rows)",
        len(per_layer_df), len(summary_df),
    )

    print("\n=== CKA Representation Similarity Summary ===")
    print(summary_df.to_string(index=False))

    if owns_model:
        del model, tokenizer, analyzer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "pair_similarities": results["pair_similarities"],
        "pair_cis": results["pair_cis"],
        "summary": summary_df,
        "per_layer": per_layer_df,
        "n_layers": results["n_layers"],
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_representation_similarity()
