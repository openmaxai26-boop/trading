"""
LSTM / GRU models for time-series prediction.
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


class LSTMNet(nn.Module):
    """Bidirectional LSTM with attention and dropout."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2,
        use_gru: bool = False,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        rnn_cls = nn.GRU if use_gru else nn.LSTM
        self.rnn = rnn_cls(
            input_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)                         # (B, T, H*2)
        attn_out, _ = self.attention(out, out, out)  # (B, T, H*2)
        out = self.norm(out + attn_out)
        out = self.dropout(out[:, -1, :])            # Last timestep
        return self.classifier(out)


class LSTMModel:
    """
    Training wrapper around LSTMNet / GRUNet.
    Supports both classification (direction) and regression (return).
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        use_gru: bool = False,
        model_name: str = "lstm",
    ) -> None:
        self.settings = settings or Settings()
        self.use_gru = use_gru
        self.model_name = model_name
        self.device = self._get_device()
        self.net: Optional[LSTMNet] = None
        self._checkpoint = os.path.join(
            self.settings.data.models_path, f"{model_name}.pt"
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

    def _build_net(self, input_dim: int) -> LSTMNet:
        cfg = self.settings.model
        return LSTMNet(
            input_dim=input_dim,
            hidden_size=cfg.lstm_hidden_size,
            num_layers=cfg.lstm_num_layers,
            dropout=cfg.lstm_dropout,
            use_gru=self.use_gru,
        ).to(self.device)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict:
        """
        Train the model.
        X: (n_samples, seq_len, n_features)
        y: (n_samples,)
        Returns training history.
        """
        cfg = self.settings.model
        input_dim = X_train.shape[2]
        self.net = self._build_net(input_dim)

        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=cfg.learning_rate * 0.01
        )
        criterion = nn.CrossEntropyLoss()

        train_loader = self._make_loader(X_train, y_train, cfg.batch_size, shuffle=True)
        val_loader = self._make_loader(X_val, y_val, cfg.batch_size, shuffle=False)

        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        best_val_acc = 0.0
        patience_counter = 0

        for epoch in range(cfg.epochs):
            # Train
            self.net.train()
            train_loss = 0.0
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.net(Xb)
                loss = criterion(logits, yb.long())
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            # Validate
            self.net.eval()
            val_loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(self.device), yb.to(self.device)
                    logits = self.net(Xb)
                    val_loss += criterion(logits, yb.long()).item()
                    preds = logits.argmax(dim=1)
                    correct += (preds == yb.long()).sum().item()
                    total += len(yb)

            val_acc = correct / total if total > 0 else 0.0
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            history["val_acc"].append(val_acc)

            scheduler.step()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self._save()
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"[{self.model_name}] Epoch {epoch+1}/{cfg.epochs} | "
                    f"Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f}"
                )

            if patience_counter >= cfg.patience:
                logger.info(f"[{self.model_name}] Early stopping at epoch {epoch+1}")
                break

        # Reload best
        self._load()
        history["best_val_acc"] = best_val_acc
        return history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns class probabilities. X: (n, seq_len, features) → (n, 2)"""
        if self.net is None:
            self._load()
        if self.net is None:
            raise RuntimeError(f"Model {self.model_name} not trained")

        self.net.eval()
        loader = self._make_loader(X, np.zeros(len(X)), batch_size=512, shuffle=False)
        probs = []
        with torch.no_grad():
            for Xb, _ in loader:
                Xb = Xb.to(self.device)
                logits = self.net(Xb)
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(probs, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns class predictions (0 or 1)."""
        return self.predict_proba(X).argmax(axis=1)

    @staticmethod
    def _make_loader(
        X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool
    ) -> DataLoader:
        tx = torch.FloatTensor(X)
        ty = torch.FloatTensor(y)
        return DataLoader(TensorDataset(tx, ty), batch_size=batch_size, shuffle=shuffle)

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self._checkpoint), exist_ok=True)
        torch.save(self.net.state_dict(), self._checkpoint)

    def _load(self) -> None:
        if os.path.exists(self._checkpoint) and self.net is not None:
            self.net.load_state_dict(torch.load(self._checkpoint, map_location=self.device))
            self.net.eval()
