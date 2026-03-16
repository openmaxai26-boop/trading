"""
Temporal Transformer model for multi-step time-series prediction.
"""

import math
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TemporalTransformerNet(nn.Module):
    """
    Transformer encoder for temporal sequences.
    Input: (batch, seq_len, input_dim)
    Output: (batch, num_classes)
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,    # Pre-norm for training stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)           # (B, T, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x)              # (B, T, d_model)
        x = self.pool(x.permute(0, 2, 1)).squeeze(-1)  # (B, d_model)
        return self.head(x)


class TransformerModel:
    """Training and inference wrapper for the temporal Transformer."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings()
        self.device = self._get_device()
        self.net: Optional[TemporalTransformerNet] = None
        self._checkpoint = os.path.join(
            self.settings.data.models_path, "transformer.pt"
        )

    def _get_device(self) -> torch.device:
        d = self.settings.model.device
        if d == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(d)

    def _build_net(self, input_dim: int) -> TemporalTransformerNet:
        cfg = self.settings.model
        return TemporalTransformerNet(
            input_dim=input_dim,
            d_model=cfg.transformer_d_model,
            nhead=cfg.transformer_nhead,
            num_encoder_layers=cfg.transformer_num_layers,
            dropout=cfg.transformer_dropout,
            max_seq_len=cfg.transformer_max_seq_len,
        ).to(self.device)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict:
        """Train the Transformer model."""
        cfg = self.settings.model
        input_dim = X_train.shape[2]
        self.net = self._build_net(input_dim)

        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        warmup_steps = 500
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.learning_rate,
            steps_per_epoch=max(1, len(X_train) // cfg.batch_size),
            epochs=cfg.epochs,
            pct_start=0.1,
        )
        criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train.astype(int))),
            batch_size=cfg.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val.astype(int))),
            batch_size=cfg.batch_size,
            shuffle=False,
        )

        history = {"train_loss": [], "val_acc": []}
        best_val_acc = 0.0
        patience_counter = 0

        for epoch in range(cfg.epochs):
            self.net.train()
            total_loss = 0.0
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.net(Xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            self.net.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(self.device), yb.to(self.device)
                    preds = self.net(Xb).argmax(dim=1)
                    correct += (preds == yb).sum().item()
                    total += len(yb)

            val_acc = correct / total if total > 0 else 0.0
            avg_loss = total_loss / len(train_loader)
            history["train_loss"].append(avg_loss)
            history["val_acc"].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self._save()
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                logger.info(f"[Transformer] Epoch {epoch+1}/{cfg.epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

            if patience_counter >= cfg.patience:
                logger.info(f"[Transformer] Early stopping at epoch {epoch+1}")
                break

        self._load()
        history["best_val_acc"] = best_val_acc
        return history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.net is None:
            self._load()
        self.net.eval()
        loader = DataLoader(
            TensorDataset(torch.FloatTensor(X)),
            batch_size=512,
            shuffle=False,
        )
        probs = []
        with torch.no_grad():
            for (Xb,) in loader:
                logits = self.net(Xb.to(self.device))
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
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
