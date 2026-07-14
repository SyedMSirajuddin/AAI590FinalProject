"""
FinBERT sentiment scoring (inference).

Satisfies the capstone deep-learning requirement together with
`finetune.py`: the transformer either comes pre-fine-tuned
(ProsusAI/finbert) or is fine-tuned locally on Financial PhraseBank.

Each headline (optionally headline + summary) is mapped to class
probabilities over {negative, neutral, positive}, and to a single
scalar `score` = P(positive) - P(negative) ∈ [-1, 1] used downstream.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import SENTIMENT

logger = logging.getLogger(__name__)


class FinBertScorer:
    """Thin wrapper around a HF sequence-classification checkpoint."""

    def __init__(self, model_name: str | None = None, device: str | None = None):
        import torch
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer)

        self.model_name = model_name or SENTIMENT.model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name).to(self.device).eval()
        # ProsusAI/finbert label order is [positive, negative, neutral];
        # resolve from the model config instead of hard-coding.
        self.id2label = {i: lab.lower()
                         for i, lab in self.model.config.id2label.items()}
        logger.info("Loaded %s on %s (labels: %s)",
                    self.model_name, self.device, self.id2label)

    @staticmethod
    def _batched(seq, n):
        for i in range(0, len(seq), n):
            yield seq[i:i + n]

    def score_texts(self, texts: list[str],
                    batch_size: int | None = None) -> pd.DataFrame:
        """Return DataFrame [p_negative, p_neutral, p_positive, score]."""
        import torch

        batch_size = batch_size or SENTIMENT.batch_size
        probs_all = []
        with torch.no_grad():
            for batch in self._batched(texts, batch_size):
                enc = self.tokenizer(batch, truncation=True, padding=True,
                                     max_length=SENTIMENT.max_length,
                                     return_tensors="pt").to(self.device)
                logits = self.model(**enc).logits
                probs_all.append(torch.softmax(logits, dim=-1).cpu().numpy())

        probs = np.vstack(probs_all)
        cols = {f"p_{self.id2label[i]}": probs[:, i] for i in range(probs.shape[1])}
        out = pd.DataFrame(cols)
        out["score"] = out["p_positive"] - out["p_negative"]
        return out

    def score_news(self, news: pd.DataFrame,
                   use_summary: bool = False) -> pd.DataFrame:
        """Score a cleaned news DataFrame; returns it with sentiment columns."""
        texts = (news["headline"] + ". " + news["summary"].fillna("")
                 if use_summary else news["headline"]).tolist()
        scores = self.score_texts(texts)
        return pd.concat([news.reset_index(drop=True), scores], axis=1)
