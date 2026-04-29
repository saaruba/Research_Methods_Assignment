# src/load_data.py
"""Load full ShareChat data from Hugging Face and save raw CSV files locally."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
from datasets import load_dataset


PLATFORMS: List[str] = ["chatgpt", "claude", "grok"]
DATASET_NAME = "tucnguyen/ShareChat"
SPLIT = "train"
RAW_DIR = Path("data/raw")


def raw_csv_paths() -> dict[str, Path]:
    """Return expected local raw CSV paths for each platform."""
    return {platform: RAW_DIR / f"{platform}.csv" for platform in PLATFORMS}


def raw_data_exists() -> bool:
    """Check whether all platform-level raw CSV files already exist."""
    return all(path.exists() for path in raw_csv_paths().values())


def load_and_save_raw_data() -> pd.DataFrame:
    """Load full train split for each platform and save as local raw CSV files."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []

    for platform in PLATFORMS:
        # Load full dataset split from Hugging Face for the target platform.
        dataset = load_dataset(DATASET_NAME, platform, split=SPLIT)
        platform_df = dataset.to_pandas()
        platform_df["platform"] = platform

        # Persist the raw dataset locally for downstream local-only processing.
        output_path = RAW_DIR / f"{platform}.csv"
        platform_df.to_csv(output_path, index=False)

        frames.append(platform_df)

    combined_df = pd.concat(frames, ignore_index=True)
    return combined_df
