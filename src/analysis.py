# src/analysis.py
"""Statistical analysis module for conversation features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, kruskal, pearsonr, shapiro
from statsmodels.stats.multicomp import pairwise_tukeyhsd


RESULTS_TABLES_DIR = Path("results/tables")


def _safe_shapiro(values: pd.Series, max_n: int = 5000) -> tuple[float, float, int] | None:
    """Run Shapiro-Wilk normality on a bounded sample to avoid oversized tests."""
    clean_values = values.dropna()
    if len(clean_values) < 3:
        return None

    sample_n = min(max_n, len(clean_values))
    sampled = clean_values.sample(n=sample_n, random_state=42) if len(clean_values) > sample_n else clean_values
    stat, p_value = shapiro(sampled)
    return float(stat), float(p_value), int(sample_n)


def _eta_squared_anova(groups: list[np.ndarray]) -> float:
    """Compute eta-squared effect size for one-way ANOVA."""
    all_values = np.concatenate(groups)
    grand_mean = float(np.mean(all_values))

    ss_between = sum(len(group) * (float(np.mean(group)) - grand_mean) ** 2 for group in groups)
    ss_total = float(np.sum((all_values - grand_mean) ** 2))

    if ss_total == 0:
        return float("nan")
    return float(ss_between / ss_total)


def run_statistical_analysis(conversation_features: pd.DataFrame) -> None:
    """Run robust descriptive/inferential statistics and export analysis artifacts."""
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Descriptive Statistics (total_messages by platform) ===")
    descriptive = (
        conversation_features.groupby("platform")["total_messages"]
        .agg(mean="mean", median="median", std="std", count="count")
        .round(3)
    )
    print(descriptive.to_string())
    descriptive.to_csv(RESULTS_TABLES_DIR / "descriptive_stats.csv")

    grouped = [
        group["total_messages"].to_numpy(dtype=float)
        for _, group in conversation_features.groupby("platform")
    ]

    statistical_lines: list[str] = []
    statistical_lines.append("ANOVA and Robust Statistical Results")
    statistical_lines.append("=" * 40)

    print("\n=== ANOVA (conversation length across platforms) ===")
    if len(grouped) < 2:
        print("Not enough platform groups to run ANOVA.")
        statistical_lines.append("ANOVA: Not enough platform groups.")
        pd.DataFrame().to_csv(RESULTS_TABLES_DIR / "tukey_results.csv", index=False)
    else:
        anova_stat, anova_p = f_oneway(*grouped)
        eta_sq = _eta_squared_anova(grouped)
        print(f"F-statistic: {anova_stat:.4f}")
        print(f"p-value: {anova_p:.6g}")
        print(f"Eta-squared: {eta_sq:.4f}")

        statistical_lines.extend(
            [
                f"ANOVA F-statistic: {anova_stat:.6f}",
                f"ANOVA p-value: {anova_p:.6g}",
                f"Eta-squared: {eta_sq:.6f}",
            ]
        )

        # Multiple tests provide complementary evidence when assumptions vary.
        print("\n=== Kruskal-Wallis (non-parametric) ===")
        kruskal_stat, kruskal_p = kruskal(*grouped)
        print(f"H-statistic: {kruskal_stat:.4f}")
        print(f"p-value: {kruskal_p:.6g}")
        statistical_lines.extend(
            [
                f"Kruskal-Wallis H-statistic: {kruskal_stat:.6f}",
                f"Kruskal-Wallis p-value: {kruskal_p:.6g}",
            ]
        )

        print("\n=== Tukey HSD Post-hoc Test ===")
        tukey = pairwise_tukeyhsd(
            endog=conversation_features["total_messages"],
            groups=conversation_features["platform"],
            alpha=0.05,
        )
        print(tukey.summary())
        statistical_lines.append("Tukey HSD Summary:")
        statistical_lines.append(str(tukey.summary()))
        tukey_df = pd.DataFrame(
            data=tukey.summary().data[1:],
            columns=tukey.summary().data[0],
        )
        tukey_df.to_csv(RESULTS_TABLES_DIR / "tukey_results.csv", index=False)

    print("\n=== Shapiro-Wilk Normality Check (sample <= 5000) ===")
    overall_shapiro = _safe_shapiro(conversation_features["total_messages"])
    if overall_shapiro is None:
        print("Overall: insufficient data for Shapiro-Wilk.")
        statistical_lines.append("Shapiro-Wilk Overall: insufficient data.")
    else:
        stat, p_value, sample_n = overall_shapiro
        print(f"Overall (n={sample_n}): W={stat:.4f}, p-value={p_value:.6g}")
        statistical_lines.append(
            f"Shapiro-Wilk Overall (n={sample_n}): W={stat:.6f}, p-value={p_value:.6g}"
        )

    for platform_name, group in conversation_features.groupby("platform"):
        platform_shapiro = _safe_shapiro(group["total_messages"])
        if platform_shapiro is None:
            print(f"{platform_name}: insufficient data for Shapiro-Wilk.")
            statistical_lines.append(f"Shapiro-Wilk {platform_name}: insufficient data.")
            continue
        stat, p_value, sample_n = platform_shapiro
        print(f"{platform_name} (n={sample_n}): W={stat:.4f}, p-value={p_value:.6g}")
        statistical_lines.append(
            f"Shapiro-Wilk {platform_name} (n={sample_n}): W={stat:.6f}, p-value={p_value:.6g}"
        )

    print("\n=== Correlation (user engagement ratio vs conversation length) ===")
    try:
        corr_stat, corr_p = pearsonr(
            conversation_features["user_ratio"],
            conversation_features["total_messages"],
        )
        print(f"Pearson r: {corr_stat:.4f}")
        print(f"p-value: {corr_p:.6g}")
        statistical_lines.extend(
            [
                f"Pearson correlation r(user_ratio, total_messages): {corr_stat:.6f}",
                f"Pearson correlation p-value: {corr_p:.6g}",
            ]
        )
    except ValueError as exc:
        print(f"Correlation could not be computed: {exc}")
        statistical_lines.append(f"Pearson correlation failed: {exc}")

    correlation_matrix = conversation_features[
        [
            "total_messages",
            "user_messages",
            "assistant_messages",
            "user_ratio",
        ]
    ].corr()
    correlation_matrix.to_csv(RESULTS_TABLES_DIR / "correlation_matrix.csv")

    (RESULTS_TABLES_DIR / "statistical_results.txt").write_text(
        "\n".join(statistical_lines), encoding="utf-8"
    )
