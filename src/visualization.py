"""Visualization module for publication-quality quantitative interaction analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Global publication-style settings for visual consistency across all figures.
sns.set_theme(style="whitegrid", context="paper")
FIG_SIZE = (8, 5)
PALETTE = sns.color_palette("Set2")
TITLE_SIZE = 14
LABEL_SIZE = 12


def create_visualizations(conversation_features: pd.DataFrame) -> None:
    """Create and save publication-quality figures."""
    figures_dir = Path("results/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Keep full data for statistics, but filter extreme values for clearer visual comparison.
    df_filtered = conversation_features[
        conversation_features["total_messages"] < 100
    ].copy()
    if df_filtered.empty:
        df_filtered = conversation_features.copy()

    # Graph 1: Clean histogram with mean reference line.
    plt.figure(figsize=FIG_SIZE)
    sns.histplot(df_filtered["total_messages"], bins=30, color=PALETTE[0])
    mean_val = float(df_filtered["total_messages"].mean())
    plt.axvline(mean_val, color="red", linestyle="--", label=f"Mean = {mean_val:.2f}")
    plt.title("Distribution of Conversation Lengths", fontsize=TITLE_SIZE)
    plt.xlabel("Total Messages", fontsize=LABEL_SIZE)
    plt.ylabel("Frequency", fontsize=LABEL_SIZE)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "hist_filtered.png", dpi=300)
    plt.close()

    # Graph 2: Clean boxplot comparison by platform.
    plt.figure(figsize=FIG_SIZE)
    sns.boxplot(
        data=df_filtered,
        x="platform",
        y="total_messages",
        palette=PALETTE,
        showfliers=False,
    )
    plt.title("Conversation Length by Platform", fontsize=TITLE_SIZE)
    plt.xlabel("Platform", fontsize=LABEL_SIZE)
    plt.ylabel("Total Messages", fontsize=LABEL_SIZE)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "boxplot_filtered.png", dpi=300)
    plt.close()

    # Graph 3: Simplified violin plot for distribution shape by platform.
    plt.figure(figsize=FIG_SIZE)
    sns.violinplot(
        data=df_filtered,
        x="platform",
        y="total_messages",
        palette=PALETTE,
        inner=None,
    )
    plt.title("Distribution of Conversation Length by Platform", fontsize=TITLE_SIZE)
    plt.xlabel("Platform", fontsize=LABEL_SIZE)
    plt.ylabel("Total Messages", fontsize=LABEL_SIZE)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "violin_plot.png", dpi=300)
    plt.close()

    # Graph 4: Sampled scatter + regression trend for readability.
    sample_n = min(2000, len(df_filtered))
    sample_df = df_filtered.sample(n=sample_n, random_state=42)
    plt.figure(figsize=FIG_SIZE)
    ax = sns.scatterplot(
        data=sample_df,
        x="total_messages",
        y="user_messages",
        hue="platform",
        palette=PALETTE,
        alpha=0.5,
        s=20,
    )
    sns.regplot(
        data=sample_df,
        x="total_messages",
        y="user_messages",
        scatter=False,
        ax=ax,
        color="black",
        line_kws={"linewidth": 1.5, "alpha": 0.9},
    )
    plt.title("User vs Total Messages", fontsize=TITLE_SIZE)
    plt.xlabel("Total Messages", fontsize=LABEL_SIZE)
    plt.ylabel("User Messages", fontsize=LABEL_SIZE)
    plt.grid(alpha=0.3)
    plt.legend(title="Platform", frameon=False, fontsize=10, title_fontsize=10)
    plt.tight_layout()
    plt.savefig(figures_dir / "scatter_regression.png", dpi=300)
    plt.close()
