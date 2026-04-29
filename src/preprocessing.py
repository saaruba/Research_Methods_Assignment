# src/preprocessing.py
"""Local-data preprocessing for conversation-level feature engineering."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RAW_FILES = ["chatgpt.csv", "claude.csv", "grok.csv"]


def load_local_raw_data() -> pd.DataFrame:
    """Load local raw CSV files and combine into one DataFrame."""
    frames: list[pd.DataFrame] = []

    for filename in RAW_FILES:
        file_path = RAW_DIR / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Missing local raw dataset file: {file_path}")

        platform_df = pd.read_csv(file_path)

        # Ensure platform exists even if CSV is altered externally.
        if "platform" not in platform_df.columns:
            platform_df["platform"] = file_path.stem

        frames.append(platform_df)

    return pd.concat(frames, ignore_index=True)


def build_conversation_features(combined_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate rows by URL to produce one feature row per conversation."""
    required_columns = {"url", "role", "platform"}
    missing_columns = required_columns.difference(combined_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required column(s): {missing}")

    # Group by URL (conversation ID) and compute message-level counts.
    conversation_features = (
        combined_df.groupby("url", as_index=False)
        .agg(
            total_messages=("role", "size"),
            user_messages=("role", lambda x: (x == "user").sum()),
            assistant_messages=("role", lambda x: (x == "llm").sum()),
            platform=("platform", "first"),
        )
        .sort_values(["platform", "url"])
        .reset_index(drop=True)
    )

    # Add interaction-level features for deeper comparative analysis.
    conversation_features["avg_turn_length"] = (
        conversation_features["total_messages"]
        / (conversation_features["user_messages"] + 1e-5)
    )
    conversation_features["user_ratio"] = (
        conversation_features["user_messages"]
        / conversation_features["total_messages"]
    )
    conversation_features["assistant_ratio"] = (
        conversation_features["assistant_messages"]
        / conversation_features["total_messages"]
    )

    # Mark outliers instead of removing them to preserve full-data transparency.
    conversation_features["is_outlier"] = conversation_features["total_messages"] > 100

    # Save processed conversation-level features locally.
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / "conversation_features.csv"
    conversation_features.to_csv(output_path, index=False)

    return conversation_features

