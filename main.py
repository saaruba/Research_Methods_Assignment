# main.py
"""Main entry point for the ShareChat research data pipeline."""

from __future__ import annotations

from src.analysis import run_statistical_analysis
from src.load_data import load_and_save_raw_data, raw_data_exists
from src.preprocessing import build_conversation_features, load_local_raw_data
from src.visualization import create_visualizations


def main() -> None:
    """Run the full local-first data analysis pipeline."""
    # Step 1: Download and save raw CSV files only if they do not exist yet.
    if not raw_data_exists():
        print("Saving raw dataset...")
        combined_df = load_and_save_raw_data()
        print(f"Saved raw files in data/raw/ ({len(combined_df)} rows total).")
    else:
        print("Raw dataset already exists in data/raw/. Skipping download.")

    # Step 2: Load local CSV files for all further processing.
    print("Loading local dataset...")
    local_df = load_local_raw_data()
    print(f"Loaded {len(local_df)} rows from local raw CSV files.")
    print(f"Dataset size (rows): {len(local_df)}")

    # Step 3: Build and save conversation-level features.
    print("Processing features...")
    conversation_features = build_conversation_features(local_df)
    print(f"Built features for {len(conversation_features)} conversations.")
    print("Saved: data/processed/conversation_features.csv")
    print(f"Number of conversations: {len(conversation_features)}")
    print(f"Number of outliers (total_messages > 100): {int(conversation_features['is_outlier'].sum())}")


    # Step 4: Run descriptive and inferential statistics.
    run_statistical_analysis(conversation_features)

    # Step 5: Generate publication-ready figures.
    create_visualizations(conversation_features)
    print("Saved figures in: results/figures/")
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
