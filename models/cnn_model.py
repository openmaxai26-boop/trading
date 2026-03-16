"""
1D CNN model for pattern recognition in time series.
"""

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)


class CNNNet(nn.Module):
    """
    Multi-scale 1D CNN with residual connections.
    Captures local patterns at multiple temporal scales.
    """

    def __init__(
        self,
        input_dim: int,
        channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        channels = channels or [64, 128, 256]

        # Multi-scale convolutions (different kernel sizes)
        self.branches = nn.ModuleList()
        for ks in [3, 5, 7]:
            branch = self._make_branch(input_dim, channels, ks, dropout)
            self.branches.append(branch)

        # Merge branches
        self.merge = nn.Sequential(
            nn.Linear(channels[-1] * 3, channels[-1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], num_classes),
        )

    def _make_branch(
        self, input_dim: int, channels: list[int], kernel_size: int, dropout: float
    ) -> nn.Sequential:
        layers = []
        in_ch = input_dim
        for out_ch in channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        layers.append(nn.AdaptiveAvgPool1d(1))  # Global avg pool
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) → (B, F, T) for Conv1d
        x = x.permute(0, 2, 1)
        branch_outs = []
        for branch in self.branches:
            out = branch(x).squeeze(-1)   # (B, channels[-1])
            branch_outs.append(out)
        merged = torch.cat(branch_outs, dim=1)  # (B, channels[-1] * 3)
        return self.merge(merged)


class CNNModel:
    """Training wrapper for 1D CNN pattern recognizer."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings()
        self.device = self._get_device()
        self.net: Optional[CNNNet] = None
        self._checkpoint = os.path.join(self.settings.data.models_path, "cnn.pt")

    def _get_device(self) -> torch.device:
        d = self.settings.model.device
        if d == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(d)

    def _build_net(self, input_dim: int) -> CNNNet:
        cfg = self.settings.model
        return CNNNet(
            input_dim=input_dim,
            channels=cfg.cnn_channels,
            kernel_size=cfg.cnn_kernel_size,
        ).to(self.device)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict:
        cfg = self.settings.model
        input_dim = X_train.shape[2]
        self.net = self._build_net(input_dim)

        optimizer = torch.optim.AdamW(self.net.parameters(), lr=cfg.learning_rate)
        criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train.astype(int))),
            batch_size=cfg.batch_size, shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val.astype(int))),
            batch_size=cfg.batch_size, shuffle=False,
        )

        best_val_acc = 0.0
        patience_counter = 0
        history: dict = {"train_loss": [], "val_acc": []}

        for epoch in range(cfg.epochs):
            self.net.train()
            total_loss = 0.0
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.net(Xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            self.net.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for Xb, yb in val_loader:
                    preds = self.net(Xb.to(self.device)).argmax(dim=1)
                    correct += (preds == yb.to(self.device)).sum().item()
                    total += len(yb)

            val_acc = correct / total if total > 0 else 0.0
            history["train_loss"].append(total_loss / len(train_loader))
            history["val_acc"].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self._save()
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                logger.info(f"[CNN] Epoch {epoch+1} | Val Acc: {val_acc:.4f}")
            if patience_counter >= cfg.patience:
                logger.info("[CNN] Early stopping")
                break

        self._load()
        history["best_val_acc"] = best_val_acc
        return history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.net is None:
            self._load()
        self.net.eval()
        loader = DataLoader(TensorDataset(torch.FloatTensor(X)), batch_size=512, shuffle=False)
        probs = []
        with torch.no_grad():
            for (Xb,) in loader:
                probs.append(torch.softmax(self.net(Xb.to(self.device)), dim=1).cpu().numpy())
        return np.concatenate(probs, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self._checkpoint), exist_ok=True)
        torch.save(self.net.state_dict(), self._checkpoint)

    def _load(self) -> None:
        if os.path.exists(self._checkpoint) and self.net is not None:
            self.net.load_state_dict(torch.load(self._checkpoint, map_location=self.device))
            self.net.eval()
