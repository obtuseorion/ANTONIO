"""
Evaluate urbas/bert_aegis on entfane/preprocessed_toxigen.

Evaluates on the `prompt` field (which prompt_label annotates directly).
Reports metrics for bert_aegis alongside the RoBERTa baseline scores
already bundled in the dataset (roberta_prediction column, tau=0.5).

Usage:
    python3 src/eval_toxigen.py
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import roc_auc_score

MODEL_ID   = "urbas/bert_aegis"
BATCH_SIZE = 32
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


class PromptDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts      = texts
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


def compute_metrics(probs: np.ndarray, labels: np.ndarray, tau: float) -> dict:
    preds = (probs >= tau).astype(int)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = (tp + tn) / len(labels)
    auc       = float(roc_auc_score(labels, probs))

    return dict(accuracy=accuracy, precision=precision, recall=recall,
                f1=f1, roc_auc=auc, tp=tp, tn=tn, fp=fp, fn=fn)


def print_metrics(name: str, metrics: dict, tau: float):
    print(f"\n  [{name}]  threshold={tau:.4f}")
    print(f"    accuracy  : {metrics['accuracy']:.4f}")
    print(f"    precision : {metrics['precision']:.4f}")
    print(f"    recall    : {metrics['recall']:.4f}")
    print(f"    F1        : {metrics['f1']:.4f}")
    print(f"    ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"    TP={metrics['tp']}  TN={metrics['tn']}  "
          f"FP={metrics['fp']}  FN={metrics['fn']}")


def main():
    # ── Dataset ───────────────────────────────────────────────────────────
    print("[data] loading entfane/preprocessed_toxigen …")
    ds = load_dataset("entfane/preprocessed_toxigen", split="train")

    texts          = [row["prompt"] for row in ds]
    labels         = np.array([row["prompt_label"]      for row in ds], dtype=int)
    roberta_probs  = np.array([row["roberta_prediction"] for row in ds], dtype=float)

    n_toxic    = int(labels.sum())
    n_nontoxic = int((labels == 0).sum())
    print(f"[data] {len(texts)} examples  |  toxic={n_toxic}  non-toxic={n_nontoxic}")

    # ── Model ─────────────────────────────────────────────────────────────
    print(f"\n[model] loading {MODEL_ID} …")
    tokenizer = BertTokenizer.from_pretrained(MODEL_ID)
    model     = BertForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval().to(DEVICE)

    tau          = float(model.config.threshold_tau)
    training_cfg = getattr(model.config, 'training_cfg', None)
    max_length   = training_cfg.get('max_length', 512) if isinstance(training_cfg, dict) else 512
    print(f"[model] threshold_tau={tau:.4f}  max_length={max_length}  device={DEVICE}")

    # ── Inference ─────────────────────────────────────────────────────────
    print("\n[inference] running …")
    dataset = PromptDataset(texts, tokenizer, max_length)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
                         pin_memory=(DEVICE == "cuda"))

    all_logits = []
    with torch.no_grad():
        for step, batch in enumerate(loader, 1):
            logits = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
            ).logits.squeeze(-1)
            all_logits.append(logits.cpu().numpy())
            if step % 20 == 0 or step == len(loader):
                print(f"  {step}/{len(loader)}", end="\r")

    print()
    probs = 1.0 / (1.0 + np.exp(-np.concatenate(all_logits)))

    # ── Metrics ───────────────────────────────────────────────────────────
    bert_m     = compute_metrics(probs,         labels, tau)
    roberta_m  = compute_metrics(roberta_probs, labels, 0.5)

    print("\n" + "=" * 60)
    print_metrics(f"bert_aegis ({MODEL_ID})", bert_m, tau)
    print_metrics("roberta_baseline  (roberta_prediction, tau=0.5)", roberta_m, 0.5)
    print("=" * 60)

    # Per-group breakdown for bert_aegis
    groups = [row["group"] for row in ds]
    unique_groups = sorted(set(groups))
    print(f"\n  [bert_aegis] per-group F1  (threshold={tau:.4f})")
    for g in unique_groups:
        idx = np.array([i for i, gr in enumerate(groups) if gr == g])
        if len(idx) == 0:
            continue
        m = compute_metrics(probs[idx], labels[idx], tau)
        print(f"    {g:<20s}  n={len(idx):>4}  F1={m['f1']:.4f}  "
              f"AUC={m['roc_auc']:.4f}  toxic={labels[idx].sum()}")


if __name__ == "__main__":
    main()
