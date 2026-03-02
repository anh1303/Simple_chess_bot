"""
train.py — Huấn luyện ChessNet
═══════════════════════════════════════════════════════
FIX so với phiên bản cũ:
  1. Input 14 kênh (match với preprocess.py mới)
  2. Thêm BatchNorm + Dropout → tránh overfit
  3. Thêm validation loop mỗi epoch
  4. Learning rate scheduler (ReduceLROnPlateau)
  5. Early stopping
  6. Lưu best model theo val_loss
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# ──────────────────────────────────────────────
# CẤU HÌNH
# ──────────────────────────────────────────────
CONFIG = {
    "data_path":    "chess-data.pt",
    "model_path":   "chess_model.pth",
    "epochs":       30,
    "batch_size":   512,
    "lr":           1e-3,
    "weight_decay": 1e-4,       # L2 regularization
    "patience":     5,           # Early stopping sau N epoch không cải thiện
    "num_workers":  0,           # Tăng lên 4 nếu chạy trên Linux/Mac
}


# ──────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────
class ResBlock(nn.Module):
    """Residual block để tránh vanishing gradient."""
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.net(x))  # skip connection


class ChessNet(nn.Module):
    """
    Architecture:
      - Stem conv: 14 → 128 kênh
      - 6 Residual blocks (128 kênh)
      - FC head: 128×8×8 → 256 → 1 (tanh)
    """
    def __init__(self, in_channels: int = 14, filters: int = 128, res_blocks: int = 6):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
        )

        # Residual tower
        self.tower = nn.Sequential(*[ResBlock(filters) for _ in range(res_blocks)])

        # Value head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.tower(x)
        return self.head(x)


# ──────────────────────────────────────────────
# TRAINING LOOP
# ──────────────────────────────────────────────
def train_model(
    data_path:  str = CONFIG["data_path"],
    model_path: str = CONFIG["model_path"],
    epochs:     int = CONFIG["epochs"],
    batch_size: int = CONFIG["batch_size"],
    lr:         float = CONFIG["lr"],
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("╔" + "═"*50 + "╗")
    print("║          CHESS MODEL TRAINING                 ║")
    print("╚" + "═"*50 + "╝")
    print(f"  Device    : {device}")
    print(f"  Data      : {data_path}")
    print(f"  Epochs    : {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  LR        : {lr}")
    print()

    # ── Load data ──
    data = torch.load(data_path, map_location="cpu")

    # Hỗ trợ cả format cũ (inputs/labels) và mới (train_inputs/val_inputs)
    if "train_inputs" in data:
        train_ds = TensorDataset(data["train_inputs"], data["train_labels"])
        val_ds   = TensorDataset(data["val_inputs"],   data["val_labels"])
    else:
        # Format cũ: tự split
        inputs = data["inputs"]
        labels = data["labels"]
        n = len(inputs)
        split = int(n * 0.9)
        idx = torch.randperm(n)
        train_ds = TensorDataset(inputs[idx[:split]], labels[idx[:split]])
        val_ds   = TensorDataset(inputs[idx[split:]], labels[idx[split:]])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=CONFIG["num_workers"], pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=CONFIG["num_workers"],
    )

    print(f"  Train size: {len(train_ds):,}")
    print(f"  Val size  : {len(val_ds):,}")
    print()

    # ── Model ──
    model     = ChessNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # Đếm params
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params    : {n_params:,}")
    print("─" * 52)
    print(f"  {'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | {'LR':>8}")
    print("─" * 52)

    best_val_loss  = float("inf")
    patience_count = 0

    for epoch in range(1, epochs + 1):

        # ── Train ──
        model.train()
        train_loss = 0.0
        for inputs_b, labels_b in train_loader:
            inputs_b, labels_b = inputs_b.to(device), labels_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs_b), labels_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs_b, labels_b in val_loader:
                inputs_b, labels_b = inputs_b.to(device), labels_b.to(device)
                val_loss += criterion(model(inputs_b), labels_b).item()
        val_loss /= len(val_loader)

        # Scheduler step
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Log
        marker = " ✅" if val_loss < best_val_loss else ""
        print(f"  {epoch:>5} | {train_loss:>10.4f} | {val_loss:>10.4f} | {current_lr:>8.2e}{marker}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_count += 1

        # Early stopping
        if patience_count >= CONFIG["patience"]:
            print(f"\n  ⏹ Early stopping tại epoch {epoch} (val loss không giảm {CONFIG['patience']} epoch)")
            break

    print("─" * 52)
    print(f"\n  ✅ Best val loss: {best_val_loss:.4f}")
    print(f"  💾 Model đã lưu vào: {model_path}")
    print("═" * 52)


if __name__ == "__main__":
    train_model(
        data_path  = CONFIG["data_path"],
        model_path = CONFIG["model_path"],
        epochs     = CONFIG["epochs"],
        batch_size = CONFIG["batch_size"],
    )