"""
Fine-tune a BERT-family model on Financial PhraseBank.

This is the *model training* element of the capstone. Two supported modes:
  1. Fine-tune `bert-base-uncased` from scratch on PhraseBank (shows the
     full training loop and lets us report train/val loss curves).
  2. Further fine-tune `ProsusAI/finbert` (domain-adaptive refinement).

Outputs:
  models/finbert-phrasebank/   — best checkpoint (by validation macro-F1)
  reports/figures/finetune_curves.png — loss/F1 curves for the report

Run:  python -m src.sentiment.finetune --base bert-base-uncased
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from src.config import PROJECT_ROOT, SENTIMENT, FIGURES_DIR
from src.data.phrasebank import LABEL2ID, ID2LABEL, load_phrasebank

logger = logging.getLogger(__name__)


def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro")}


def main(base_model: str | None = None):
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer, Trainer, TrainingArguments,
                              set_seed)

    set_seed(SENTIMENT.seed)
    base_model = base_model or SENTIMENT.finetune_base
    splits = load_phrasebank(seed=SENTIMENT.seed)

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True,
                         max_length=SENTIMENT.max_length)

    ds = {name: Dataset.from_pandas(df).map(tokenize, batched=True)
          for name, df in splits.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=3,
        id2label=ID2LABEL, label2id=LABEL2ID)

    out_dir = PROJECT_ROOT / "models" / "finbert-phrasebank"
    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=SENTIMENT.epochs,
        per_device_train_batch_size=SENTIMENT.batch_size,
        per_device_eval_batch_size=SENTIMENT.batch_size * 2,
        learning_rate=SENTIMENT.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=25,
        report_to="none",
        seed=SENTIMENT.seed,
    )

    trainer = Trainer(model=model, args=args,
                      train_dataset=ds["train"], eval_dataset=ds["val"],
                      processing_class=tokenizer,
                      compute_metrics=compute_metrics)
    trainer.train()

    # Held-out test evaluation — quote these numbers in the report.
    test_metrics = trainer.evaluate(ds["test"], metric_key_prefix="test")
    logger.info("Test metrics: %s", test_metrics)

    trainer.save_model(str(out_dir / "best"))
    tokenizer.save_pretrained(str(out_dir / "best"))

    _plot_curves(trainer.state.log_history)


def _plot_curves(log_history: list[dict]):
    """Save train-loss / val-F1 curves for the report's Results section."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train = [(h["step"], h["loss"]) for h in log_history if "loss" in h]
    evals = [(h["step"], h["eval_macro_f1"])
             for h in log_history if "eval_macro_f1" in h]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    if train:
        ax1.plot(*zip(*train), label="train loss", color="tab:blue")
        ax1.set_ylabel("Loss"); ax1.set_xlabel("Step")
    if evals:
        ax2 = ax1.twinx()
        ax2.plot(*zip(*evals), label="val macro-F1",
                 color="tab:orange", marker="o")
        ax2.set_ylabel("Macro-F1")
    fig.suptitle("FinBERT fine-tuning on Financial PhraseBank")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "finetune_curves.png", dpi=150)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=None,
                        help="Base checkpoint to fine-tune")
    main(parser.parse_args().base)
