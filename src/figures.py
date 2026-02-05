"""Figure generation for the analysis report."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

STYLE_CONFIG = {
    "font.size": 12,
    "axes.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "legend.frameon": False,
    "figure.dpi": 300,
    "font.family": "sans-serif",
}

COLORS = {
    "en": "#264653",
    "ja": "#E76F51",
    "ko": "#F4A261",
    "zh": "#2A9D8F",
    "vi": "#E9C46A",
    "id": "#606C38",
    "th": "#8338EC",
    "ar": "#D46F4D",
    "de": "#FFBF66",
    "fr": "#457B9D",
}

PAIR_COLORS = [
    "#264653", "#E76F51", "#2A9D8F", "#F4A261",
    "#E9C46A", "#606C38", "#8338EC", "#457B9D",
]


def _setup_style():
    plt.rcParams.update(STYLE_CONFIG)


def plot_tokenizer_fertility(
    fertility_df: pd.DataFrame,
    output_path: str,
    reference_values: Optional[Dict[str, float]] = None,
) -> str:
    """Generate tokenizer fertility bar chart."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    df = fertility_df.sort_values("mean_fertility", ascending=True)
    languages = df["language"].values
    lang_names = df.get("language_name", df["language"]).values
    fertility = df["mean_fertility"].values
    errors = df["std_fertility"].values

    colors = [COLORS.get(lang, "#999999") for lang in languages]

    bars = ax.barh(
        range(len(languages)), fertility, xerr=errors,
        color=colors, edgecolor="white", linewidth=0.5,
        capsize=3, error_kw={"linewidth": 1},
    )

    ax.set_yticks(range(len(languages)))
    ax.set_yticklabels(lang_names, fontsize=11)
    ax.set_xlabel("Token Fertility", fontsize=13, fontweight="bold")
    ax.set_title("Aya Expanse 8B: Token Fertility by Language", fontsize=14, fontweight="bold")

    if reference_values:
        for label, value in reference_values.items():
            ax.axvline(x=value, color="#999999", linestyle="--", linewidth=1, alpha=0.7)
            ax.text(value + 0.02, len(languages) - 0.5, label, fontsize=9, color="#666666")

    for i, (v, e) in enumerate(zip(fertility, errors)):
        ax.text(v + e + 0.02, i, f"{v:.2f}", va="center", fontsize=10)

    ax.set_xlim(0, max(fertility) * 1.3)
    ax.grid(axis="x", alpha=0.2, linestyle="--")

    plt.tight_layout()
    save_path = Path(output_path) / "figures" / "tokenizer_fertility.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)
    return str(save_path)


def plot_logit_lens_heatmap(
    layer_matrix_df: pd.DataFrame,
    output_path: str,
) -> str:
    """Generate logit lens heatmap."""
    _setup_style()

    pivot = layer_matrix_df.pivot_table(
        values="top1_prob", index="language", columns="layer", aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        pivot, ax=ax, cmap="YlOrRd", vmin=0, vmax=1,
        linewidths=0.3, linecolor="white",
        cbar_kws={"label": "Top-1 Probability", "shrink": 0.8},
    )

    ax.set_xlabel("Layer", fontsize=13, fontweight="bold")
    ax.set_ylabel("Language", fontsize=13, fontweight="bold")
    ax.set_title(
        "Aya Expanse 8B: Logit Lens -- Top-1 Prediction Confidence by Layer",
        fontsize=14, fontweight="bold",
    )

    plt.tight_layout()
    save_path = Path(output_path) / "figures" / "logit_lens_heatmap.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)
    return str(save_path)


def plot_language_gap(
    summary_df: pd.DataFrame,
    output_path: str,
) -> str:
    """Generate language gap bar chart."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    agg = summary_df.groupby("language_name").agg(
        mean_emergence=("emergence_layer", "mean"),
        mean_crystallization=("crystallization_layer", "mean"),
        mean_gap=("language_gap", "mean"),
    ).reset_index().dropna(subset=["mean_gap"])

    agg = agg.sort_values("mean_gap", ascending=True)

    x = np.arange(len(agg))
    width = 0.35

    lang_codes = {v: k for k, v in {
        "en": "English", "ja": "Japanese", "ko": "Korean",
        "zh": "Chinese", "vi": "Vietnamese", "id": "Indonesian", "th": "Thai",
    }.items()}

    colors_emergence = [COLORS.get(lang_codes.get(n, ""), "#264653") for n in agg["language_name"]]
    colors_gap = [COLORS.get(lang_codes.get(n, ""), "#E76F51") for n in agg["language_name"]]

    ax.bar(x - width / 2, agg["mean_emergence"], width, label="Concept Emergence Layer",
           color=colors_emergence, alpha=0.7, edgecolor="white")
    ax.bar(x + width / 2, agg["mean_gap"], width, label="Language Gap (extra layers)",
           color=colors_gap, alpha=0.9, edgecolor="white")

    ax.set_xlabel("Language", fontsize=13, fontweight="bold")
    ax.set_ylabel("Layer Count", fontsize=13, fontweight="bold")
    ax.set_title("Concept Emergence vs Language-Specific Crystallization Gap", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(agg["language_name"], fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.2, linestyle="--")

    plt.tight_layout()
    save_path = Path(output_path) / "figures" / "language_gap.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)
    return str(save_path)


def plot_feature_overlap_curves(
    overlap_results: Dict[str, Dict],
    output_path: str,
) -> str:
    """Generate per-layer Jaccard similarity curves."""
    _setup_style()
    n_models = len(overlap_results)
    fig, axes = plt.subplots(1, max(n_models, 1), figsize=(8 * max(n_models, 1), 6), squeeze=False)

    for model_idx, (model_num, results) in enumerate(overlap_results.items()):
        ax = axes[0, model_idx]
        pair_sims = results.get("pair_similarities", {})
        n_layers = results.get("n_layers", 12)

        for pair_idx, (label, sims) in enumerate(pair_sims.items()):
            color = PAIR_COLORS[pair_idx % len(PAIR_COLORS)]
            ax.plot(
                range(len(sims)), sims,
                marker="o", linewidth=2, markersize=5,
                color=color, markerfacecolor=color,
                markeredgecolor="white", markeredgewidth=0.8,
                label=label,
            )

        ax.set_xlabel("Layer", fontsize=13, fontweight="bold")
        ax.set_ylabel("Jaccard Similarity", fontsize=13, fontweight="bold")
        ax.set_title(f"Model: GPT2 Multilingual {model_num}%", fontsize=14, fontweight="bold")
        ax.set_xlim(-0.5, n_layers - 0.5)
        ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.2, linestyle="--")
        ax.legend(fontsize=8, loc="upper right")

    plt.suptitle(
        "Cross-Lingual Feature Overlap by Layer",
        fontsize=16, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    save_path = Path(output_path) / "figures" / "feature_overlap_curves.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)
    return str(save_path)


def plot_intervention_success(
    summary_df: pd.DataFrame,
    output_path: str,
) -> str:
    """Generate intervention success matrix heatmap."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    if summary_df.empty:
        ax.text(0.5, 0.5, "No intervention results available",
                transform=ax.transAxes, ha="center", va="center", fontsize=14)
        ax.set_axis_off()
    else:
        experiments = summary_df["experiment"].values
        success_rates = summary_df["success_rate"].values * 100

        colors = ["#2A9D8F" if r > 20 else "#E76F51" if r > 0 else "#999999" for r in success_rates]

        bars = ax.barh(range(len(experiments)), success_rates, color=colors, edgecolor="white")
        ax.set_yticks(range(len(experiments)))
        ax.set_yticklabels(experiments, fontsize=10)
        ax.set_xlabel("Success Rate (%)", fontsize=13, fontweight="bold")
        ax.set_xlim(0, 100)

        for i, v in enumerate(success_rates):
            ax.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=10)

    ax.set_title("Language Switching Intervention Success Rates", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.2, linestyle="--")

    plt.tight_layout()
    save_path = Path(output_path) / "figures" / "intervention_success.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)
    return str(save_path)


def plot_multilingual_entropy(
    entropy_df: pd.DataFrame,
    output_path: str,
) -> str:
    """Generate multilingual entropy curve."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    layers = entropy_df["layer"].values
    entropy = entropy_df["entropy"].values

    ax.plot(layers, entropy, marker="o", linewidth=2.5, markersize=7,
            color="#264653", markerfacecolor="#264653",
            markeredgecolor="white", markeredgewidth=1.2)

    ax.fill_between(layers, entropy, alpha=0.1, color="#264653")

    ax.set_xlabel("Layer", fontsize=13, fontweight="bold")
    ax.set_ylabel("Prediction Entropy (bits)", fontsize=13, fontweight="bold")
    ax.set_title(
        "Multilingual Prediction Entropy by Layer (Aya Expanse 8B)",
        fontsize=14, fontweight="bold",
    )

    n_layers = len(layers)
    if n_layers > 6:
        ax.axvspan(0, n_layers * 0.2, alpha=0.05, color="#E76F51", label="Early: Language encoding")
        ax.axvspan(n_layers * 0.2, n_layers * 0.6, alpha=0.05, color="#2A9D8F", label="Middle: Shared space")
        ax.axvspan(n_layers * 0.6, n_layers, alpha=0.05, color="#F4A261", label="Late: Language decoding")
        ax.legend(fontsize=10, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    ax.grid(alpha=0.2, linestyle="--")

    plt.tight_layout()
    save_path = Path(output_path) / "figures" / "multilingual_entropy.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)
    return str(save_path)


def plot_representation_similarity(
    cka_results: Dict[str, Any],
    output_path: str,
) -> str:
    """Generate two-panel CKA representation similarity figure."""
    _setup_style()

    cjk_pairs = ["EN-EN", "JA-EN", "KO-EN", "ZH-EN", "JA-ZH", "JA-KO"]
    latin_pairs = ["EN-EN", "JA-JA", "EN-VI", "EN-ID"]

    pair_similarities = cka_results.get("pair_similarities", {})
    pair_cis = cka_results.get("pair_cis", {})
    n_layers = cka_results.get("n_layers", 32)

    pair_color_map = {
        "EN-EN": "#264653",
        "JA-JA": "#E76F51",
        "JA-EN": "#E76F51",
        "KO-EN": "#F4A261",
        "ZH-EN": "#2A9D8F",
        "JA-ZH": "#8338EC",
        "JA-KO": "#D46F4D",
        "EN-VI": "#E9C46A",
        "EN-ID": "#606C38",
    }

    pair_linestyle = {
        "EN-EN": "--",
        "JA-JA": "--",
    }

    fig, (ax_cjk, ax_latin) = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    layers = np.arange(n_layers)

    def _plot_panel(ax, pair_list, title):
        for label in pair_list:
            if label not in pair_similarities:
                continue
            ckas = pair_similarities[label]
            color = pair_color_map.get(label, "#999999")
            ls = pair_linestyle.get(label, "-")
            lw = 1.5 if ls == "--" else 2.0

            ax.plot(
                layers[:len(ckas)], ckas,
                linewidth=lw, linestyle=ls, color=color, label=label,
            )

            if label in pair_cis:
                cis = pair_cis[label]
                ci_lower = [c[0] for c in cis]
                ci_upper = [c[1] for c in cis]
                ax.fill_between(
                    layers[:len(ci_lower)], ci_lower, ci_upper,
                    alpha=0.12, color=color,
                )

        early_end = int(n_layers * 0.2)
        late_start = int(n_layers * 0.625)
        ax.axvspan(0, early_end, alpha=0.04, color="#E76F51")
        ax.axvspan(late_start, n_layers, alpha=0.04, color="#F4A261")

        ax.text(
            early_end / 2, 0.02, "Early", ha="center", fontsize=10,
            color="#333333", fontweight="bold",
        )
        ax.text(
            (early_end + late_start) / 2, 0.02, "Middle", ha="center",
            fontsize=10, color="#333333", fontweight="bold",
        )
        ax.text(
            (late_start + n_layers) / 2, 0.02, "Late", ha="center",
            fontsize=10, color="#333333", fontweight="bold",
        )

        ax.set_xlabel("Layer", fontsize=13, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlim(-0.5, n_layers - 0.5)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.2, linestyle="--")
        ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1.02, 0.5))

    _plot_panel(ax_cjk, cjk_pairs, "CJK Language Pairs")
    _plot_panel(ax_latin, latin_pairs, "Latin-Script Controls")

    ax_cjk.set_ylabel("CKA Similarity", fontsize=13, fontweight="bold")

    plt.suptitle(
        "Cross-Lingual Representation Similarity (CKA, Aya Expanse 8B)",
        fontsize=15, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    save_path = Path(output_path) / "figures" / "representation_similarity.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)
    return str(save_path)


def generate_all_figures(
    tokenizer_results: Optional[Dict] = None,
    logit_lens_results: Optional[Dict] = None,
    cka_results: Optional[Dict] = None,
    overlap_results: Optional[Dict] = None,
    switching_results: Optional[Dict] = None,
    output_dir: str = "outputs",
) -> List[str]:
    """Generate all figures from available analysis results."""
    output_path = Path(output_dir)
    figures = []

    if tokenizer_results and "fertility" in tokenizer_results:
        path = plot_tokenizer_fertility(
            tokenizer_results["fertility"], str(output_path),
            reference_values={"EN (CLT paper)": 1.53, "AR (CLT paper)": 1.97},
        )
        figures.append(path)

    if logit_lens_results:
        if "layer_matrix" in logit_lens_results:
            path = plot_logit_lens_heatmap(logit_lens_results["layer_matrix"], str(output_path))
            figures.append(path)

        if "summary" in logit_lens_results:
            path = plot_language_gap(logit_lens_results["summary"], str(output_path))
            figures.append(path)

        if "entropy" in logit_lens_results:
            path = plot_multilingual_entropy(logit_lens_results["entropy"], str(output_path))
            figures.append(path)

    if cka_results and "pair_similarities" in cka_results:
        path = plot_representation_similarity(cka_results, str(output_path))
        figures.append(path)

    if overlap_results and "overlap_results" in overlap_results:
        path = plot_feature_overlap_curves(overlap_results["overlap_results"], str(output_path))
        figures.append(path)

    if switching_results and "summary" in switching_results:
        path = plot_intervention_success(switching_results["summary"], str(output_path))
        figures.append(path)

    logger.info("Generated %d figures", len(figures))
    return figures
