"""
Financial PhraseBank loader (Malo et al., 2014).

The corpus contains ~4,800 sentences from financial news, labeled
positive / negative / neutral at four annotator-agreement levels.
We default to `sentences_75agree`.

datasets>=5.0 removed script-based loading and this HF repo has no
parquet branch, so we download the original corpus zip from the repo
and parse it directly (lines are 'sentence@label', latin-1 encoded).
"""

from __future__ import annotations

import zipfile

import pandas as pd
from sklearn.model_selection import train_test_split

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# Maps HF config names to files inside the official corpus zip
_SPLIT_FILES = {
    "sentences_50agree":  "Sentences_50Agree.txt",
    "sentences_66agree":  "Sentences_66Agree.txt",
    "sentences_75agree":  "Sentences_75Agree.txt",
    "sentences_allagree": "Sentences_AllAgree.txt",
}


def load_phrasebank(split: str = "sentences_75agree",
                    seed: int = 42) -> dict[str, pd.DataFrame]:
    """Load Financial PhraseBank and produce a stratified 80/10/10
    train/val/test split with integer labels per LABEL2ID."""
    from huggingface_hub import hf_hub_download

    zip_path = hf_hub_download("takala/financial_phrasebank",
                               "data/FinancialPhraseBank-v1.0.zip",
                               repo_type="dataset")
    fname = f"FinancialPhraseBank-v1.0/{_SPLIT_FILES[split]}"
    with zipfile.ZipFile(zip_path) as zf, zf.open(fname) as f:
        lines = f.read().decode("latin-1").splitlines()

    texts, labels = [], []
    for line in lines:
        if "@" not in line:
            continue
        sentence, _, label = line.rpartition("@")
        texts.append(sentence.strip())
        labels.append(LABEL2ID[label.strip().lower()])
    df = pd.DataFrame({"text": texts, "label": labels})

    train, rest = train_test_split(df, test_size=0.20, random_state=seed,
                                   stratify=df["label"])
    val, test = train_test_split(rest, test_size=0.50, random_state=seed,
                                 stratify=rest["label"])
    return {"train": train.reset_index(drop=True),
            "val": val.reset_index(drop=True),
            "test": test.reset_index(drop=True)}