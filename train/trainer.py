"""
train/trainer.py
Offline LSTM training script.
Reads dataset.json exported from the browser capture tool,
trains an LSTM classifier, and exports to kendama.onnx.

Usage:
    pip install torch numpy scikit-learn
    python train/trainer.py --data data/dataset.json --out models/kendama.onnx
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ── Constants (must match pipeline/labels.js) ──
TRICK_IDS   = ['spike', 'big_cup', 'lighthouse', 'around_japan', 'airplane', 'bird']
NUM_CLASSES = len(TRICK_IDS)
FEATURE_DIM = 113    # 33*3 + 8 + 6
WINDOW_SIZE = 30


# ── Dataset ──

class KendamaDataset(Dataset):
    """
    Slices each clip into overlapping windows of length WINDOW_SIZE.
    Each window gets the label of the containing clip.
    """
    def __init__(self, clips, stride=5):
        self.samples = []
        for clip in clips:
            frames = np.array(clip['frames'], dtype=np.float32)  # (T, FEATURE_DIM)
            label  = TRICK_IDS.index(clip['label'])
            T = len(frames)
            if T < WINDOW_SIZE:
                # Pad with zeros
                pad = np.zeros((WINDOW_SIZE - T, FEATURE_DIM), dtype=np.float32)
                frames = np.vstack([pad, frames])
                T = WINDOW_SIZE
            for start in range(0, T - WINDOW_SIZE + 1, stride):
                window = frames[start : start + WINDOW_SIZE]
                self.samples.append((window, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y, dtype=torch.long)


# ── Model ──

class KendamaLSTM(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, hidden_dim=128, num_layers=2,
                 num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        out    = self.dropout(out[:, -1, :])   # last timestep
        return self.fc(out)                     # (batch, num_classes)


# ── Training ──

def train(args):
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # Load dataset
    with open(args.data) as f:
        clips = json.load(f)
    print(f"Loaded {len(clips)} clips  |  labels: {set(c['label'] for c in clips)}")

    # Train/val split at clip level (not frame level, to avoid leakage)
    train_clips, val_clips = train_test_split(clips, test_size=0.2, random_state=42,
                                               stratify=[c['label'] for c in clips])

    train_ds = KendamaDataset(train_clips, stride=5)
    val_ds   = KendamaDataset(val_clips,   stride=WINDOW_SIZE)  # no overlap for eval

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)

    print(f"Train windows: {len(train_ds)}  |  Val windows: {len(val_ds)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model     = KendamaLSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        train_loss, correct, total = 0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
            correct    += (logits.argmax(1) == yb).sum().item()
            total      += len(xb)
        scheduler.step()

        # ── Val ──
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        all_preds, all_true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits  = model(xb)
                loss    = criterion(logits, yb)
                val_loss   += loss.item() * len(xb)
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_total   += len(xb)
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_true.extend(yb.cpu().numpy())

        tr_acc  = correct    / total     * 100
        val_acc = val_correct / val_total * 100
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"loss={train_loss/total:.4f}  acc={tr_acc:.1f}%  "
              f"val_loss={val_loss/val_total:.4f}  val_acc={val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), '/tmp/best_kendama.pt')

    # Reload best checkpoint
    model.load_state_dict(torch.load('/tmp/best_kendama.pt', map_location=device))
    model.eval()

    print(f"\nBest val accuracy: {best_val_acc:.1f}%")
    print("\nClassification report (val):")
    print(classification_report(all_true, all_preds, target_names=TRICK_IDS))

    # ── Export to ONNX ──
    dummy  = torch.zeros(1, WINDOW_SIZE, FEATURE_DIM).to(device)
    onnx_path = args.out
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        opset_version=14,
        do_constant_folding=True
    )
    print(f"\nModel exported to {onnx_path}")


# ── CLI ──

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train KendamaCV LSTM')
    parser.add_argument('--data',   default='data/dataset.json', help='Path to dataset.json')
    parser.add_argument('--out',    default='models/kendama.onnx', help='Output .onnx path')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch',  type=int, default=32)
    parser.add_argument('--lr',     type=float, default=1e-3)
    args = parser.parse_args()
    train(args)