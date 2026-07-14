"""
data_loader.py
--------------
Load the OneStopEnglish corpus into a pandas DataFrame.
"""

import os
import pandas as pd
from pathlib import Path


def load_onestop_corpus(corpus_path: str) -> pd.DataFrame:
    """
    Load the OneStopEnglish corpus from the cloned GitHub repository.

    Args:
        corpus_path: Path to the 'Texts-SeparatedByReadingLevel' folder.

    Returns:
        A DataFrame with columns: ['text', 'level', 'label', 'filename']
    """
    level_map = {
        "Ele-Txt": {"level": "Elementary", "label": 0},
        "Int-Txt": {"level": "Intermediate", "label": 1},
        "Adv-Txt": {"level": "Advanced", "label": 2},
    }

    records = []

    for folder_name, meta in level_map.items():
        folder_path = Path(corpus_path) / folder_name
        if not folder_path.exists():
            print(f"Warning: {folder_path} not found. Skipping.")
            continue

        for file in sorted(folder_path.glob("*.txt")):
            text = file.read_text(encoding="utf-8", errors="ignore").strip()
            if text:  # skip empty files
                records.append({
                    "text": text,
                    "level": meta["level"],
                    "label": meta["label"],
                    "filename": file.name,
                })

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} texts across {df['level'].nunique()} levels.")
    print(f"Distribution:\n{df['level'].value_counts().to_string()}")
    return df


if __name__ == "__main__":
    # Adjust this path to match your local setup
    corpus_path = r"C:\Users\46729\Downloads\nlp-text-complexity-analyzer\nlp-text-complexity-analyzer\data\raw\OneStopEnglishCorpus\Texts-SeparatedByReadingLevel"
    df = load_onestop_corpus(corpus_path)
    print(df.head())