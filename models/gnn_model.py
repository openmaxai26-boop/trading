"""
Graph Neural Network for modeling relationships between assets.
Uses PyTorch Geometric (torch_geometric).
"""

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    from torch_geometric.nn import GATConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    GEO_AVAILABLE = True
except ImportError:
    GEO_AVAILABLE = False
    logger.warning("torch_geometric not available. GNN will use fallback MLP.")


class GNNNet(nn.Module):
    """
    Graph Attention Network for multi-asset relationship modeling.
    Each asset is a node; edges represent correlation relationships.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_channels: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.use_gnn = GEO_AVAILABLE
        self.dropout = dropout

        if GEO_AVAILABLE:
            self.convs = nn.ModuleList()
            self.convs.append(GATConv(input_dim, hidden_channels, heads=4, dropout=dropout))
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_channels * 4, hidden_channels, heads=4, dropout=dropout))
            self.convs.append(GATConv(hidden_channels * 4, hidden_channels, heads=1, dropout=dropout))
            self.head = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels // 2, num_classes),
            )
        else:
            # MLP fallback
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, num_classes),
            )

    def forward(self, x: torch.Tensor, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.use_gnn or edge_index is None:
            # Fallback: treat as tabular input (mean over sequence if 3D)
            if x.dim() == 3:
                x = x.mean(dim=1)
            return self.mlp(x)

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return self.head(x)


class GNNModel:
    """
    GNN for cross-asset modeling.
    Builds correlation graphs from a universe of assets.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings()
        self.device = self._get_device()
        self.net: Optional[GNNNet] = None
        self._checkpoint = os.path.join(self.settings.data.models_path, "gnn.pt")

    def _get_device(self) -> torch.device:
        d = self.settings.model.device
        if d == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(d)

    def build_correlation_graph(
        self,
        returns_matrix: np.ndarray,
        correlation_threshold: float = 0.5,
    ) -> Optional[torch.Tensor]:
        """
        Build edge_index from a correlation matrix.
        Only keep edges where |correlation| > threshold.
        Returns edge_index tensor of shape (2, n_edges).
        """
        if not GEO_AVAILABLE:
            return None

        n_assets = returns_matrix.shape[0]
        corr = np.corrcoef(returns_matrix)
        src, dst = [], []
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                if abs(corr[i, j]) > correlation_threshold:
                    src += [i, j]
                    dst += [j, i]

        if not src:
            return None

        edge_index = torch.tensor([src, dst], dtype=torch.long).to(self.device)
        return edge_index

    def predict_proba_single(
        self, node_features: np.ndarray, edge_index: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Single-asset prediction given node features and graph structure.
        node_features: (n_assets, feature_dim)
        Returns probabilities for the target asset (index 0).
        """
        if self.net is None:
            self._load()
        if self.net is None:
            # Return uniform if no model
            return np.array([[0.5, 0.5]])

        self.net.eval()
        x = torch.FloatTensor(node_features).to(self.device)
        with torch.no_grad():
            logits = self.net(x, edge_index)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs[:1]  # Return target asset prediction

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self._checkpoint), exist_ok=True)
        if self.net:
            torch.save(self.net.state_dict(), self._checkpoint)

    def _load(self) -> None:
        if os.path.exists(self._checkpoint):
            logger.info(f"Loading GNN from {self._checkpoint}")
