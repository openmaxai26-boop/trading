"""
Temporal embedding generation via LSTM autoencoder.
Learns compressed representations of multi-variate time series.
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)


class LSTMAutoEncoder(nn.Module):
    """LSTM-based autoencoder for temporal sequence compression."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout
        )
        self.encoder_proj = nn.Linear(hidden_dim, embedding_dim)

        # Decoder
        self.decoder_proj = nn.Linear(embedding_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout
        )
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_dim) → embedding: (batch, embedding_dim)"""
        _, (h, _) = self.encoder_lstm(x)
        return self.encoder_proj(h[-1])  # Use last hidden state

    def decode(self, embedding: torch.Tensor, seq_len: int) -> torch.Tensor:
        """embedding: (batch, embedding_dim) → reconstructed: (batch, seq_len, input_dim)"""
        h = self.decoder_proj(embedding).unsqueeze(0)  # (1, batch, hidden)
        # Repeat embedding across time steps
        dec_input = h.permute(1, 0, 2).expand(-1, seq_len, -1)
        out, _ = self.decoder_lstm(dec_input, (h, torch.zeros_like(h)))
        return self.output_proj(out)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encode(x)
        reconstructed = self.decode(embedding, x.size(1))
        return embedding, reconstructed


class TemporalEmbeddingGenerator:
    """
    Trains and uses an LSTM autoencoder to generate embeddings
    from multi-variate time series windows.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings()
        self.embedding_dim = self.settings.features.embedding_dim
        self.hidden_dim = self.settings.features.encoder_hidden_dim
        self.seq_len = self.settings.features.sequence_length
        self.model: Optional[LSTMAutoEncoder] = None
        self.device = self._get_device()
        self.feature_columns: list[str] = []
        self._checkpoint_path = os.path.join(
            self.settings.data.models_path, "temporal_encoder.pt"
        )

    def _get_device(self) -> torch.device:
        device_str = self.settings.model.device
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device_str)

    def _build_model(self, input_dim: int) -> LSTMAutoEncoder:
        model = LSTMAutoEncoder(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            embedding_dim=self.embedding_dim,
        ).to(self.device)
        return model

    def _prepare_sequences(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """Extract numeric features and build sliding window sequences."""
        # Select numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Drop columns with too many NaNs
        valid_cols = [c for c in numeric_cols if df[c].isna().mean() < 0.2]
        data = df[valid_cols].ffill().fillna(0).values.astype(np.float32)

        # Normalize per feature (z-score)
        mean = data.mean(axis=0)
        std = data.std(axis=0) + 1e-8
        data = (data - mean) / std

        # Build sequences
        seqs = []
        for i in range(len(data) - self.seq_len + 1):
            seqs.append(data[i: i + self.seq_len])

        return np.array(seqs), valid_cols

    def train(self, df: pd.DataFrame, epochs: int = 50, batch_size: int = 64) -> None:
        """Train the autoencoder on historical data."""
        seqs, self.feature_columns = self._prepare_sequences(df)
        if len(seqs) < batch_size:
            logger.warning("Not enough sequences to train temporal encoder")
            return

        input_dim = seqs.shape[2]
        self.model = self._build_model(input_dim)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        tensor = torch.FloatTensor(seqs).to(self.device)
        loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=True)

        self.model.train()
        best_loss = float("inf")
        for epoch in range(epochs):
            total_loss = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                _, reconstructed = self.model(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            if avg_loss < best_loss:
                best_loss = avg_loss
                self._save_model()

            if (epoch + 1) % 10 == 0:
                logger.info(f"Autoencoder epoch {epoch+1}/{epochs} — loss: {avg_loss:.6f}")

        logger.info(f"Temporal encoder trained. Best loss: {best_loss:.6f}")

    def generate_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate embeddings for each timestep (using most recent window).
        Returns DataFrame with embedding_dim columns aligned to df index.
        """
        if self.model is None:
            if os.path.exists(self._checkpoint_path):
                self._load_model(df)
            else:
                logger.warning("No trained temporal encoder; returning empty embeddings")
                return pd.DataFrame(index=df.index)

        seqs, _ = self._prepare_sequences(df)
        if len(seqs) == 0:
            return pd.DataFrame(index=df.index)

        self.model.eval()
        tensor = torch.FloatTensor(seqs).to(self.device)
        with torch.no_grad():
            batch_size = 512
            all_embeddings = []
            for i in range(0, len(tensor), batch_size):
                batch = tensor[i: i + batch_size]
                emb = self.model.encode(batch)
                all_embeddings.append(emb.cpu().numpy())

        embeddings = np.concatenate(all_embeddings, axis=0)

        # Align to df index (sequences start at seq_len-1 index)
        idx = df.index[self.seq_len - 1:]
        cols = [f"emb_{i}" for i in range(self.embedding_dim)]
        emb_df = pd.DataFrame(embeddings, index=idx[:len(embeddings)], columns=cols)
        return emb_df.reindex(df.index)

    def _save_model(self) -> None:
        if self.model is None:
            return
        os.makedirs(os.path.dirname(self._checkpoint_path), exist_ok=True)
        torch.save(
            {"state_dict": self.model.state_dict(), "feature_columns": self.feature_columns},
            self._checkpoint_path,
        )

    def _load_model(self, df: pd.DataFrame) -> None:
        try:
            checkpoint = torch.load(self._checkpoint_path, map_location=self.device)
            self.feature_columns = checkpoint.get("feature_columns", [])
            input_dim = len(self.feature_columns)
            if input_dim > 0:
                self.model = self._build_model(input_dim)
                self.model.load_state_dict(checkpoint["state_dict"])
                self.model.eval()
                logger.info(f"Temporal encoder loaded from {self._checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to load temporal encoder: {e}")
