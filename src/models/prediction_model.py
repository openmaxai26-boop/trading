"""
MODÈLE DE PRÉDICTION HYBRIDE  (Couche 4)
==========================================

ARCHITECTURE : LSTM + Transformer + CNN combinés

POURQUOI UN MODÈLE HYBRIDE ?
Chaque architecture capture un type différent de pattern :

┌────────────────┬──────────────────────────────────────────────────┐
│ Architecture   │ Ce qu'elle capture                               │
├────────────────┼──────────────────────────────────────────────────┤
│ LSTM           │ Dépendances temporelles séquentielles             │
│                │ → "Le marché a monté pendant 5 jours de suite"   │
├────────────────┼──────────────────────────────────────────────────┤
│ Transformer    │ Relations à longue distance (attention)           │
│                │ → "Ce pattern ressemble à celui d'il y a 3 mois" │
├────────────────┼──────────────────────────────────────────────────┤
│ CNN            │ Patterns locaux / formes de bougies               │
│                │ → "C'est un double creux" ou "épaule-tête"       │
└────────────────┴──────────────────────────────────────────────────┘

SORTIE PROBABILISTE :
Le modèle ne dit PAS "le prix sera à 185$".
Il dit "il y a 72% de probabilité que le prix MONTE".
C'est beaucoup plus honnête et utile pour la gestion des risques.

ARCHITECTURE DÉTAILLÉE :
                 Input (batch, seq_len, n_features)
                        │
          ┌─────────────┼────────────┐
          ▼             ▼            ▼
       [LSTM]    [Transformer]    [CNN]
          │             │            │
          └──── Concat (sum) ────────┘
                        │
                  [Linear 256]
                        │
                   [Dropout]
                        │
                  [Linear 128]
                        │
                   [Dropout]
                        │
              [Linear → 2 sorties]
              [mu (moyenne), log_var (variance)]
                        │
              Distribution Normale → Probabilité de hausse
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional, List
from pathlib import Path

from ..utils.logger import get_logger

logger = get_logger("PredictionModel")


# ══════════════════════════════════════════════════════════════════
#  BLOCS DE CONSTRUCTION DU RÉSEAU
# ══════════════════════════════════════════════════════════════════

class LSTMBlock(nn.Module):
    """
    Bloc LSTM (Long Short-Term Memory).

    QU'EST-CE QUE LE LSTM ?
    Un réseau récurrent avec une "mémoire" à long terme.
    Il peut retenir des informations importantes pendant
    de longues séquences et oublier ce qui n'est pas pertinent.

    Inventé par Hochreiter & Schmidhuber (1997).
    Standard de l'industrie pour les séries temporelles.

    PARAMÈTRES :
    - input_size  : Nombre de features en entrée
    - hidden_size : Taille de la mémoire interne (128 neurones)
    - num_layers  : Couches LSTM empilées (2 = plus profond)
    - dropout     : Probabilité de désactivation aléatoire (régularisation)
    """

    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,   # (batch, seq, features) → plus intuitif
            bidirectional=False
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passage avant.
        x : (batch, seq_len, input_size)
        → (batch, hidden_size)  ← on prend uniquement le dernier timestep
        """
        out, _ = self.lstm(x)       # out: (batch, seq_len, hidden_size)
        out = self.norm(out)
        return out[:, -1, :]        # Dernier timestep : (batch, hidden_size)


class TransformerBlock(nn.Module):
    """
    Bloc Transformer avec mécanisme d'attention multi-tête.

    QU'EST-CE QUE LE TRANSFORMER ?
    Introduit par Vaswani et al. (2017) — la même architecture
    qui alimente GPT et ChatGPT.

    LE MÉCANISME D'ATTENTION :
    Pour chaque jour de la séquence, le modèle calcule l'importance
    (l'"attention") de tous les autres jours.

    EXEMPLE :
    Pour prédire demain, le modèle peut apprendre que :
    "Ce pattern du jour d'aujourd'hui ressemble à ce qui s'est
    passé il y a 30 jours → regarder ce jour-là attentivement"

    PARAMÈTRES :
    - d_model     : Dimension du modèle (doit être divisible par nhead)
    - nhead       : Nombre de têtes d'attention (8 têtes en parallèle)
    - num_layers  : Couches d'encodeur empilées
    - dropout     : Dropout pour la régularisation
    """

    def __init__(self, input_size: int, d_model: int,
                 nhead: int, num_layers: int, dropout: float):
        super().__init__()
        # Projection de l'entrée vers d_model
        self.input_proj = nn.Linear(input_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True     # Pre-LN : plus stable à l'entraînement
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, seq_len, input_size)
        → (batch, d_model)
        """
        x = self.input_proj(x)          # Projeter vers d_model
        x = self.transformer(x)         # Attention sur toute la séquence
        x = self.norm(x)
        return x[:, -1, :]              # Dernier token : (batch, d_model)


class CNNBlock(nn.Module):
    """
    Bloc CNN (Convolutional Neural Network) pour patterns locaux.

    POURQUOI CNN POUR LES SÉRIES TEMPORELLES ?
    Les CNN appliquent des filtres glissants qui détectent
    des patterns locaux (sur une fenêtre de K jours).

    ANALOGIE :
    Un filtre de taille 3 regarde 3 jours consécutifs.
    Il peut apprendre à reconnaître des bougies comme :
    - "Marteau" (corps bas, mèche haute longue)
    - "Étoile filante" (corps haut, mèche basse longue)
    - "Doji" (ouverture ≈ clôture)

    PARAMÈTRES :
    - input_size  : Nombre de features
    - channels    : Liste de canaux [32, 64, 128]
    - kernel_sizes: Tailles des filtres [3, 5, 7]
    """

    def __init__(self, input_size: int, channels: List[int],
                 kernel_sizes: List[int]):
        super().__init__()

        layers = []
        in_ch = input_size

        for out_ch, k in zip(channels, kernel_sizes):
            # Conv1d : (batch, channels, seq_len) — transposer l'entrée
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=k,
                                    padding=k // 2))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.1))
            in_ch = out_ch

        self.conv_layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)  # → (batch, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, seq_len, input_size)
        → (batch, channels[-1])
        """
        x = x.transpose(1, 2)       # (batch, input_size, seq_len)
        x = self.conv_layers(x)      # (batch, channels[-1], seq_len)
        x = self.pool(x)             # (batch, channels[-1], 1)
        return x.squeeze(-1)         # (batch, channels[-1])


# ══════════════════════════════════════════════════════════════════
#  MODÈLE HYBRIDE COMPLET
# ══════════════════════════════════════════════════════════════════

class HybridPredictionModel(nn.Module):
    """
    Modèle hybride LSTM + Transformer + CNN pour la prédiction de marché.

    UTILISATION :
        model = HybridPredictionModel(n_features=45, config=config)
        mu, log_var = model(x_batch)          # Sortie probabiliste
        proba_hausse = model.predict_proba(x) # 0.0 → 1.0

    SORTIE PROBABILISTE :
    Le modèle prédit une DISTRIBUTION des rendements futurs :
    - mu      : Rendement moyen prédit
    - log_var : Log-variance (incertitude)

    La probabilité de hausse = P(rendement > 0) sous cette distribution.
    """

    def __init__(self, n_features: int, config: dict):
        super().__init__()
        self.n_features = n_features
        cfg = config["prediction_model"]

        lstm_cfg   = cfg["lstm"]
        trans_cfg  = cfg["transformer"]
        cnn_cfg    = cfg["cnn"]

        # ── Branches parallèles ──
        self.lstm_block = LSTMBlock(
            input_size=n_features,
            hidden_size=lstm_cfg["hidden_size"],
            num_layers=lstm_cfg["num_layers"],
            dropout=lstm_cfg["dropout"]
        )

        self.transformer_block = TransformerBlock(
            input_size=n_features,
            d_model=trans_cfg["d_model"],
            nhead=trans_cfg["nhead"],
            num_layers=trans_cfg["num_encoder_layers"],
            dropout=trans_cfg["dropout"]
        )

        self.cnn_block = CNNBlock(
            input_size=n_features,
            channels=cnn_cfg["channels"],
            kernel_sizes=cnn_cfg["kernel_sizes"]
        )

        # ── Taille de la couche de fusion ──
        fusion_size = (lstm_cfg["hidden_size"] +
                       trans_cfg["d_model"] +
                       cnn_cfg["channels"][-1])

        # ── Tête de régression probabiliste ──
        self.head = nn.Sequential(
            nn.Linear(fusion_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        # Deux sorties : moyenne (mu) et log-variance
        self.out_mu      = nn.Linear(128, 1)
        self.out_log_var = nn.Linear(128, 1)

        # Initialisation des poids
        self._init_weights()

    def _init_weights(self):
        """Initialisation He pour les couches linéaires (meilleure convergence)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Passage avant du modèle.

        PARAMÈTRE :
        - x : Tenseur d'entrée (batch_size, seq_len, n_features)

        RETOURNE :
        - mu      : Rendement moyen prédit    (batch_size, 1)
        - log_var : Log-variance              (batch_size, 1)
        """
        # Chaque branche traite la séquence indépendamment
        lstm_out  = self.lstm_block(x)         # (batch, hidden_size)
        trans_out = self.transformer_block(x)  # (batch, d_model)
        cnn_out   = self.cnn_block(x)          # (batch, channels[-1])

        # Concaténation des sorties des 3 branches
        combined = torch.cat([lstm_out, trans_out, cnn_out], dim=-1)

        # Tête de prédiction
        h = self.head(combined)

        mu      = self.out_mu(h)
        log_var = self.out_log_var(h).clamp(-10, 2)  # Limiter pour stabilité

        return mu, log_var

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convertit la sortie probabiliste en probabilité de hausse.

        FORMULE :
        P(hausse) = P(rendement > 0)
                  = 1 - Φ(-mu / sigma)
        où Φ est la fonction de répartition normale standard.

        RETOURNE : Probabilité ∈ [0, 1] que le prix monte.
        """
        with torch.no_grad():
            mu, log_var = self.forward(x)
            sigma = torch.exp(0.5 * log_var).clamp(min=1e-6)
            # P(X > 0) sous N(mu, sigma²)
            prob = torch.sigmoid(mu / (sigma + 1e-6))
        return prob

    def get_signal(self, x: torch.Tensor) -> Dict:
        """
        Génère un signal de trading structuré.

        RETOURNE :
        {
            "signal"      : "BUY" / "SELL" / "HOLD"
            "probability" : 0.0 → 100.0  (%)
            "confidence"  : 0.0 → 100.0  (%)
            "risk_level"  : "LOW" / "MEDIUM" / "HIGH"
            "mu"          : Rendement attendu
            "sigma"       : Incertitude (écart-type)
        }
        """
        with torch.no_grad():
            mu, log_var = self.forward(x)
            sigma = torch.exp(0.5 * log_var)
            prob  = self.predict_proba(x)

        mu_val    = float(mu.mean())
        sigma_val = float(sigma.mean())
        prob_val  = float(prob.mean())

        # Signal basé sur la probabilité de hausse
        if prob_val > 0.60:
            signal = "BUY"
        elif prob_val < 0.40:
            signal = "SELL"
        else:
            signal = "HOLD"

        # Niveau de risque basé sur l'incertitude (sigma)
        if sigma_val < 0.01:
            risk = "LOW"
        elif sigma_val < 0.025:
            risk = "MEDIUM"
        else:
            risk = "HIGH"

        # Confiance = certitude de la prédiction (distance à 50%)
        confidence = abs(prob_val - 0.5) * 200.0  # 0 → 100%

        return {
            "signal"           : signal,
            "probability_up"   : round(prob_val * 100, 1),
            "confidence_score" : round(confidence, 1),
            "risk_level"       : risk,
            "expected_return"  : round(mu_val * 100, 3),
            "uncertainty"      : round(sigma_val * 100, 3)
        }


# ══════════════════════════════════════════════════════════════════
#  ENTRAÎNEUR DU MODÈLE
# ══════════════════════════════════════════════════════════════════

class ModelTrainer:
    """
    Gère l'entraînement, la validation et la sauvegarde du modèle.

    CYCLE D'ENTRAÎNEMENT :
    Pour chaque époque :
      1. TRAIN  : Parcourir tous les mini-lots, calculer la perte,
                  rétropropager les gradients, mettre à jour les poids
      2. VAL    : Évaluer sur les données de validation (sans gradient)
      3. Si pas d'amélioration pendant N époques → Early Stopping

    QU'EST-CE QUE L'EARLY STOPPING ?
    Si le modèle n'améliore pas ses performances sur la validation
    pendant 15 époques de suite, on arrête l'entraînement.
    → Évite le surapprentissage (overfitting)

    FONCTION DE PERTE : Negative Log-Likelihood (NLL)
    NLL = log(sigma) + (y - mu)² / (2 × sigma²)
    → Pénalise DEUX choses : la mauvaise prédiction ET la sur-confiance.
    """

    def __init__(
        self,
        model: HybridPredictionModel,
        config: dict,
        device: Optional[str] = None
    ):
        self.model  = model
        self.config = config
        train_cfg   = config["prediction_model"]["training"]

        # Détecter automatiquement GPU ou CPU
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)

        logger.info(f"Entraînement sur : {self.device}")

        # Optimiseur Adam : adapte le taux d'apprentissage automatiquement
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=1e-5   # Régularisation L2
        )

        # Scheduler : réduit le LR si pas d'amélioration
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5,
            patience=5
        )

        self.epochs   = train_cfg["epochs"]
        self.patience = train_cfg["patience"]
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.train_history: List[Dict] = []

    def _gaussian_nll_loss(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Perte Negative Log-Likelihood gaussienne.

        FORMULE :
        NLL = 0.5 × [log(2π) + log_var + (y - mu)² / exp(log_var)]

        EN PRATIQUE :
        - Si mu prédit correctement y et log_var est petit → perte faible
        - Si mu est loin de y → perte grande (mauvaise prédiction)
        - Si log_var est grand (grande incertitude) → perte grande aussi
          (le modèle ne doit pas "tricher" en disant juste "je suis incertain")
        """
        var = torch.exp(log_var).clamp(min=1e-6)
        loss = 0.5 * (log_var + (y.unsqueeze(-1) - mu).pow(2) / var)
        return loss.mean()

    def train_epoch(self, loader: DataLoader) -> float:
        """Entraîne le modèle sur une époque complète."""
        self.model.train()
        total_loss = 0.0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Remise à zéro des gradients
            self.optimizer.zero_grad()

            # Passage avant
            mu, log_var = self.model(X_batch)

            # Calcul de la perte
            loss = self._gaussian_nll_loss(mu, log_var, y_batch)

            # Rétropropagation (calcul des gradients)
            loss.backward()

            # Gradient clipping : évite les explosions de gradient
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Mise à jour des poids
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / max(len(loader), 1)

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> float:
        """Évalue le modèle sur les données de validation."""
        self.model.eval()
        total_loss = 0.0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            mu, log_var = self.model(X_batch)
            loss = self._gaussian_nll_loss(mu, log_var, y_batch)
            total_loss += loss.item()

        return total_loss / max(len(loader), 1)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model_path: str = "./models/prediction_model.pt"
    ) -> List[Dict]:
        """
        Entraîne le modèle complet avec early stopping.

        PARAMÈTRES :
        - train_loader : DataLoader des données d'entraînement
        - val_loader   : DataLoader des données de validation
        - model_path   : Chemin de sauvegarde du meilleur modèle

        RETOURNE :
        - Historique d'entraînement (loss par époque)
        """
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Début de l'entraînement ({self.epochs} époques max)…")
        logger.info(f"Paramètres : {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss   = self.validate(val_loader)

            self.scheduler.step(val_loss)

            epoch_stats = {
                "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss
            }
            self.train_history.append(epoch_stats)

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), model_path)
            else:
                self.patience_counter += 1

            # Log toutes les 10 époques
            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    f"  Époque {epoch:3d}/{self.epochs} | "
                    f"Train={train_loss:.4f} | Val={val_loss:.4f} | "
                    f"Patience={self.patience_counter}/{self.patience}"
                )

            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping à l'époque {epoch} ✓")
                break

        # Charger le meilleur modèle
        if Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"Meilleur modèle rechargé (val_loss={self.best_val_loss:.4f}) ✓")

        return self.train_history

    def evaluate(self, test_loader: DataLoader) -> Dict:
        """
        Évalue les performances finales sur les données de TEST.

        MÉTRIQUES :
        - Directional Accuracy : % de fois où la direction est bonne
          (prédire hausse quand ça monte, baisse quand ça descend)
        - MAE : Erreur absolue moyenne sur les rendements
        """
        self.model.eval()
        all_mu, all_y = [], []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                mu, _ = self.model(X_batch)
                all_mu.append(mu.cpu().numpy().flatten())
                all_y.append(y_batch.numpy())

        if not all_mu:
            return {}

        preds  = np.concatenate(all_mu)
        actual = np.concatenate(all_y)

        # Accuracy directionnelle
        dir_acc = np.mean(np.sign(preds) == np.sign(actual)) * 100
        mae     = np.mean(np.abs(preds - actual)) * 100

        results = {
            "directional_accuracy": round(dir_acc, 2),
            "mae_pct"             : round(mae, 4),
            "n_samples"           : len(actual)
        }

        logger.info("─" * 45)
        logger.info("RÉSULTATS SUR DONNÉES DE TEST :")
        logger.info(f"  Accuracy directionnelle : {dir_acc:.1f}%")
        logger.info(f"  MAE rendement           : {mae:.4f}%")
        logger.info(f"  Échantillons évalués    : {len(actual)}")
        logger.info("─" * 45)

        return results
