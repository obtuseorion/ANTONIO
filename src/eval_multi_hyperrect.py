import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, GPT2Config
from sklearn.cluster import DBSCAN, AgglomerativeClustering


CYBER_DATASET_ID   = "urbas/cyber_harm_llama"
TOXIGEN_DATASET_ID = "entfane/preprocessed_toxigen"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_classifier(checkpoint_path: str, gpt2_variant: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    model = GPT2ForSequenceClassification.from_pretrained(
        gpt2_variant,
        num_labels=1,
        ignore_mismatched_sizes=True,
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    tokenizer = GPT2Tokenizer.from_pretrained(gpt2_variant)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts   = texts
        self.labels  = labels
        self.tok     = tokenizer
        self.max_len = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_hf_dataset(dataset_id):
    from datasets import load_dataset as hf_load
    ds = hf_load(dataset_id)
    split = ds["train"] if "train" in ds else list(ds.values())[0]
    cols = split.column_names
    if "text" in cols:
        texts = list(split["text"])
    elif "user" in cols and "assistant" in cols:
        texts = [u.strip() + "\n" + a.strip()
                 for u, a in zip(split["user"], split["assistant"])]
    elif "prompt" in cols and "generation" in cols:
        texts = [p.strip() + "\n" + g.strip()
                 for p, g in zip(split["prompt"], split["generation"])]
    else:
        raise ValueError(f"Unrecognised columns: {cols}")
    labels = list(split["label"] if "label" in cols else split["prompt_label"])
    return texts, labels


def load_all_samples(which: str) -> tuple:
    """Load and merge one or both datasets into a single (texts, labels) pair."""
    texts, labels = [], []
    if which in ("cyber", "both"):
        t, l = load_hf_dataset(CYBER_DATASET_ID)
        texts  += t
        labels += l
        print(f"  [cyber_harm_llama]     {len(t)} samples")
    if which in ("toxigen", "both"):
        t, l = load_hf_dataset(TOXIGEN_DATASET_ID)
        texts  += t
        labels += l
        print(f"  [preprocessed_toxigen] {len(t)} samples")
    return texts, labels


# ─────────────────────────────────────────────────────────────────────────────
# 3. Extract last hidden states
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_last_hidden_states(
    model: GPT2ForSequenceClassification,
    dataloader: DataLoader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    all_hidden, all_labels = [], []

    for batch in dataloader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].numpy()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        last_layer     = outputs.hidden_states[-1]
        last_token_idx = attention_mask.sum(dim=1) - 1
        batch_size     = last_layer.size(0)
        idx = last_token_idx.view(batch_size, 1, 1).expand(
            batch_size, 1, last_layer.size(-1)
        )
        cls_hidden = last_layer.gather(1, idx).squeeze(1)

        all_hidden.append(cls_hidden.cpu().float().numpy())
        all_labels.append(labels)

    return np.vstack(all_hidden), np.concatenate(all_labels)


# ─────────────────────────────────────────────────────────────────────────────
# 4. SVD rotation
# ─────────────────────────────────────────────────────────────────────────────

def compute_svd_rotation(X: np.ndarray) -> np.ndarray:
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt


# ─────────────────────────────────────────────────────────────────────────────
# 5. Clustering
# ─────────────────────────────────────────────────────────────────────────────

def cluster_embeddings(X: np.ndarray, min_cluster_size: int, method: str = "agglomerative") -> np.ndarray:
    """
    Cluster embeddings using hierarchical clustering.

    Args:
        X: Embeddings array of shape (n_samples, n_features)
        min_cluster_size: Minimum samples required to form a cluster
        method: Clustering method ("agglomerative" or "dbscan")

    Returns:
        cluster_labels: Array of cluster labels (-1 for noise in DBSCAN)
    """
    if method == "agglomerative":
        # Use distance threshold to control minimum cluster size
        n_clusters = max(2, len(X) // min_cluster_size)
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward'
        )
        labels = clustering.fit_predict(X)

        # Filter out small clusters
        unique_labels, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            if count < min_cluster_size:
                labels[labels == label] = -1

    elif method == "dbscan":
        # Estimate eps based on min_cluster_size
        from sklearn.neighbors import NearestNeighbors
        neighbors = NearestNeighbors(n_neighbors=min_cluster_size)
        neighbors_fit = neighbors.fit(X)
        distances, _ = neighbors_fit.kneighbors(X)
        eps = np.median(distances[:, -1])

        clustering = DBSCAN(eps=eps, min_samples=min_cluster_size)
        labels = clustering.fit_predict(X)
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    return labels


# ─────────────────────────────────────────────────────────────────────────────
# 6. Multiple Hyper-rectangles
# ─────────────────────────────────────────────────────────────────────────────

class MultipleHyperRectangles:
    def __init__(self, seed_hidden: np.ndarray, min_cluster_size: int, method: str = "agglomerative"):
        """
        Build multiple hyperrectangles by clustering the seed data.

        Args:
            seed_hidden: Harmful samples to build rectangles from
            min_cluster_size: Minimum samples per cluster
            method: Clustering method
        """
        self.min_cluster_size = min_cluster_size
        self.method = method

        # Compute global SVD rotation
        self.Vt = compute_svd_rotation(seed_hidden)
        rotated = seed_hidden @ self.Vt.T

        # Cluster in rotated space
        self.cluster_labels = cluster_embeddings(rotated, min_cluster_size, method)

        # Build one hyperrectangle per cluster
        self.rectangles = []
        unique_labels = np.unique(self.cluster_labels)
        unique_labels = unique_labels[unique_labels >= 0]  # Remove noise label (-1)

        for label in unique_labels:
            cluster_mask = self.cluster_labels == label
            cluster_points = rotated[cluster_mask]

            if len(cluster_points) >= min_cluster_size:
                lower = cluster_points.min(axis=0)
                upper = cluster_points.max(axis=0)
                self.rectangles.append((lower, upper))

        self.n_clusters = len(self.rectangles)
        self.r = seed_hidden.shape[1]

    def predict(self, hidden: np.ndarray) -> np.ndarray:
        """
        Predict if samples fall inside ANY of the hyperrectangles.
        """
        rotated = hidden @ self.Vt.T
        predictions = np.zeros(len(hidden), dtype=int)

        for lower, upper in self.rectangles:
            inside = np.all(
                (rotated >= lower) & (rotated <= upper),
                axis=1,
            )
            predictions = np.logical_or(predictions, inside)

        return predictions.astype(int)

    def print_diagnostics(self, seed_hidden: np.ndarray, all_hidden: np.ndarray):
        all_rot = all_hidden @ self.Vt.T
        full_widths = all_rot.max(axis=0) - all_rot.min(axis=0) + 1e-12

        total_volume = 0.0
        for lower, upper in self.rectangles:
            box_widths = upper - lower
            volume = float(np.exp(
                np.mean(np.log(np.maximum(box_widths, 1e-12) / full_widths))
            ))
            total_volume += volume

        n_noise = int((self.cluster_labels == -1).sum())
        n_clustered = int((self.cluster_labels >= 0).sum())

        print(f"  Hidden dim (d)           : {all_hidden.shape[1]}")
        print(f"  SVD rank used (r)        : {self.r}")
        print(f"  Min cluster size         : {self.min_cluster_size}")
        print(f"  Number of clusters       : {self.n_clusters}")
        print(f"  Clustered samples        : {n_clustered}/{len(seed_hidden)}")
        print(f"  Noise samples (excluded) : {n_noise}/{len(seed_hidden)}")
        print(f"  Total volume coverage    : {total_volume:.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                precision=precision, recall=recall, f1=f1)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Multiple Hyper-Rectangle Evaluation with Clustering"
    )
    p.add_argument("--checkpoint",       default="gpt2_classifier.pt")
    p.add_argument("--gpt2_variant",     default="gpt2")
    p.add_argument("--datasets",         default="both",
                   choices=["cyber", "toxigen", "both"])
    p.add_argument("--n-eval",           type=int, default=100,
                   help="Number of eval samples per class")
    p.add_argument("--min-cluster-values", type=int, nargs='+',
                   help="Specific min_cluster values to test (e.g., 2 5 10 20 50 100)")
    p.add_argument("--min-cluster-start", type=int, default=2,
                   help="Start value for min_cluster sweep")
    p.add_argument("--min-cluster-end", type=int, default=150,
                   help="End value for min_cluster sweep")
    p.add_argument("--min-cluster-step", type=int, default=10,
                   help="Step size for min_cluster sweep")
    p.add_argument("--clustering-method", default="agglomerative",
                   choices=["agglomerative", "dbscan"])
    p.add_argument("--seed",             type=int, default=42)
    p.add_argument("--device",           default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    p.add_argument("--max_length",       type=int, default=512)
    p.add_argument("--batch_size",       type=int, default=8)
    p.add_argument("--out",              default="multi_hyperrect_results.json")
    return p.parse_args()


def run_single_evaluation(
    seed_hidden: np.ndarray,
    test_hidden: np.ndarray,
    test_labels: np.ndarray,
    all_hidden: np.ndarray,
    min_cluster_size: int,
    method: str,
) -> dict:
    """Run evaluation for a single min_cluster_size value."""
    spec = MultipleHyperRectangles(seed_hidden, min_cluster_size, method)

    y_pred = spec.predict(test_hidden)
    y_true = test_labels
    metrics = compute_metrics(y_true, y_pred)

    return {
        "min_cluster_size": min_cluster_size,
        "n_clusters": spec.n_clusters,
        "n_clustered": int((spec.cluster_labels >= 0).sum()),
        "n_noise": int((spec.cluster_labels == -1).sum()),
        **metrics
    }


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Load model ───────────────────────────────────────────────────────────
    print(f"[1/4] Loading finetuned classifier: {args.checkpoint}")
    model, tokenizer = load_classifier(args.checkpoint, args.gpt2_variant, args.device)
    cfg: GPT2Config = model.config
    print(f"      Architecture : {args.gpt2_variant}  |  hidden_dim={cfg.n_embd}")

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"\n[2/4] Loading dataset(s) ...")
    texts, labels = load_all_samples(args.datasets)
    labels_np = np.array(labels)
    n_harm   = int((labels_np == 1).sum())
    n_benign = int((labels_np == 0).sum())
    print(f"      Total: {len(texts)}  |  harmful: {n_harm}  benign: {n_benign}")

    # ── Extract hidden states ────────────────────────────────────────────────
    print(f"\n[3/4] Extracting last hidden states ...")
    ds     = TextDataset(texts, labels, tokenizer, args.max_length)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    hidden, lbls = extract_last_hidden_states(model, loader, args.device)
    print(f"      Hidden state matrix: {hidden.shape}")

    # ── Split ────────────────────────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    harm_idx   = np.where(lbls == 1)[0]
    benign_idx = np.where(lbls == 0)[0]
    rng.shuffle(harm_idx)
    rng.shuffle(benign_idx)

    test_harm_idx = harm_idx[:args.n_eval]
    test_ben_idx  = benign_idx[:args.n_eval]
    seed_harm_idx = harm_idx[args.n_eval:]

    seed_hidden = hidden[seed_harm_idx]
    test_hidden = np.vstack([hidden[test_harm_idx], hidden[test_ben_idx]])
    test_labels = np.concatenate([
        np.ones(len(test_harm_idx),  dtype=int),
        np.zeros(len(test_ben_idx),  dtype=int),
    ])

    print(f"\n      Seed harmful  : {len(seed_hidden)}")
    print(f"      Test harmful  : {len(test_harm_idx)}")
    print(f"      Test benign   : {len(test_ben_idx)}")
    print(f"      Test total    : {len(test_hidden)}")

    # ── Determine min_cluster values to test ─────────────────────────────────
    if args.min_cluster_values:
        min_cluster_values = sorted(args.min_cluster_values)
    else:
        min_cluster_values = list(range(
            args.min_cluster_start,
            args.min_cluster_end + 1,
            args.min_cluster_step
        ))

    print(f"\n[4/4] Running evaluations for {len(min_cluster_values)} min_cluster values...")
    print(f"      Values: {min_cluster_values}")
    print(f"      Clustering method: {args.clustering_method}")

    # ── Run evaluations ──────────────────────────────────────────────────────
    all_results = []

    for i, min_cluster in enumerate(min_cluster_values, 1):
        print(f"\n  [{i}/{len(min_cluster_values)}] min_cluster_size = {min_cluster}")

        result = run_single_evaluation(
            seed_hidden, test_hidden, test_labels, hidden,
            min_cluster, args.clustering_method
        )
        all_results.append(result)

        print(f"      Clusters: {result['n_clusters']}  "
              f"Clustered: {result['n_clustered']}/{len(seed_hidden)}  "
              f"Noise: {result['n_noise']}")
        print(f"      Precision: {result['precision']:.4f}  "
              f"Recall: {result['recall']:.4f}  "
              f"F1: {result['f1']:.4f}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  MULTIPLE HYPER-RECTANGLE EVALUATION SUMMARY")
    print("=" * 90)
    print(f"{'MinCluster':>11} {'Clusters':>9} {'Clustered':>10} {'Precision':>10} "
          f"{'Recall':>10} {'F1':>10}")
    print("-" * 90)

    for r in all_results:
        print(f"{r['min_cluster_size']:>11} {r['n_clusters']:>9} "
              f"{r['n_clustered']:>10} {r['precision']:>10.4f} "
              f"{r['recall']:>10.4f} {r['f1']:>10.4f}")

    print("=" * 90)

    # ── Save results ─────────────────────────────────────────────────────────
    output = {
        "dataset": args.datasets,
        "backbone": args.gpt2_variant,
        "hidden_dim": int(cfg.n_embd),
        "n_seed_harmful": len(seed_hidden),
        "n_test": len(test_hidden),
        "clustering_method": args.clustering_method,
        "results": all_results,
    }

    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved -> {args.out}")
    return output


if __name__ == "__main__":
    main()
