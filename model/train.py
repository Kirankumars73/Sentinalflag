"""
SentinelFlag — Model Trainer
Fine-tunes DistilBERT as a 5-class sensitivity classifier.

Requirements:
  pip install torch transformers datasets scikit-learn pandas tqdm

For GPU (recommended):
  pip install torch --index-url https://download.pytorch.org/whl/cu118

Run:
  python train.py

Output:
  ./checkpoints/best_model/   — best checkpoint (by val F1)
  ./checkpoints/final_model/  — final model after all epochs
  training_log.json           — epoch-by-epoch metrics
"""

import os
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR.parent / "data" / "output"
CKPT_DIR    = BASE_DIR / "checkpoints"
LOG_FILE    = BASE_DIR / "training_log.json"

MODEL_NAME  = "distilbert-base-uncased"
NUM_LABELS  = 5
MAX_LEN     = 128          # tokens — covers 95%+ of our samples
BATCH_SIZE  = 32           # reduce to 16 if OOM on GPU
EPOCHS      = 5
LR          = 2e-5
WARMUP_RATIO= 0.1
WEIGHT_DECAY= 0.01
SEED        = 42

LABEL_NAMES = ["non-sensitive", "PII", "financial", "health", "location"]

CKPT_DIR.mkdir(parents=True, exist_ok=True)

# ── Reproducibility ───────────────────────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ── Dataset ───────────────────────────────────────────────────────────────
class SentinelDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len):
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["text", "label"])
        self.texts  = df["text"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Evaluation helper ─────────────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0

    with torch.no_grad():
        for batch in loader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            lbls  = batch["label"].to(device)

            out  = model(input_ids=ids, attention_mask=mask, labels=lbls)
            total_loss += out.loss.item()
            preds = out.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(lbls.cpu().numpy())

    acc  = accuracy_score(all_labels, all_preds)
    f1   = f1_score(all_labels, all_preds, average="macro")
    loss = total_loss / len(loader)
    return acc, f1, loss, all_preds, all_labels


# ── Main training loop ────────────────────────────────────────────────────
def train():
    print("\n=== SentinelFlag Model Training ===\n")

    # Tokenizer
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    # Datasets
    print("Loading datasets...")
    train_ds = SentinelDataset(DATA_DIR / "train.csv", tokenizer, MAX_LEN)
    val_ds   = SentinelDataset(DATA_DIR / "val.csv",   tokenizer, MAX_LEN)
    test_ds  = SentinelDataset(DATA_DIR / "test.csv",  tokenizer, MAX_LEN)

    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # Model
    print(f"\nLoading model: {MODEL_NAME} ({NUM_LABELS} classes)")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label={i: n for i, n in enumerate(LABEL_NAMES)},
        label2id={n: i for i, n in enumerate(LABEL_NAMES)},
    )
    model.to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    # Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = GradScaler() if DEVICE.type == "cuda" else None

    print(f"\nTraining config:")
    print(f"  Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LR}")
    print(f"  Steps: {total_steps} | Warmup: {warmup_steps}")
    print()

    training_log = []
    best_val_f1  = 0.0
    best_epoch   = 0

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", ncols=90)

        for step, batch in enumerate(pbar):
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            lbls = batch["label"].to(DEVICE)

            optimizer.zero_grad()

            if scaler:
                with autocast():
                    out  = model(input_ids=ids, attention_mask=mask, labels=lbls)
                    loss = out.loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                out  = model(input_ids=ids, attention_mask=mask, labels=lbls)
                loss = out.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        val_acc, val_f1, val_loss, _, _ = evaluate(model, val_loader, DEVICE)
        epoch_time = time.time() - epoch_start

        print(f"\nEpoch {epoch}: train_loss={avg_train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | val_F1={val_f1:.4f} "
              f"[{epoch_time:.0f}s]")

        entry = {
            "epoch":          epoch,
            "train_loss":     round(avg_train_loss, 4),
            "val_loss":       round(val_loss, 4),
            "val_accuracy":   round(val_acc, 4),
            "val_f1_macro":   round(val_f1, 4),
            "epoch_seconds":  round(epoch_time, 1),
        }
        training_log.append(entry)

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            best_path   = CKPT_DIR / "best_model"
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)
            print(f"  ★ New best model saved (val_F1={val_f1:.4f}) → {best_path}")

    # Save final model
    final_path = CKPT_DIR / "final_model"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    # Save training log
    with open(LOG_FILE, "w") as f:
        json.dump(training_log, f, indent=2)
    print(f"\nTraining log → {LOG_FILE}")

    # ── Final test evaluation ─────────────────────────────────────────────
    print("\n=== Final Test Evaluation (best model) ===\n")
    best_model = DistilBertForSequenceClassification.from_pretrained(CKPT_DIR / "best_model")
    best_model.to(DEVICE)

    test_acc, test_f1, test_loss, preds, labels = evaluate(best_model, test_loader, DEVICE)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test macro F1: {test_f1:.4f}")
    print(f"Best epoch:    {best_epoch}")
    print()
    print(classification_report(labels, preds, target_names=LABEL_NAMES, digits=4))

    print("Confusion matrix:")
    cm = confusion_matrix(labels, preds)
    header = f"{'':>16}" + "".join(f"{n:>14}" for n in LABEL_NAMES)
    print(header)
    for i, row in enumerate(cm):
        print(f"{LABEL_NAMES[i]:>16}" + "".join(f"{v:>14}" for v in row))

    print(f"\n✓ Training complete. Best model at: {CKPT_DIR / 'best_model'}")
    print("Next step: run  python export_onnx.py")


if __name__ == "__main__":
    train()
