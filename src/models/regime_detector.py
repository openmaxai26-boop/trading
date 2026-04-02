"""
DÉTECTEUR DE RÉGIMES DE MARCHÉ  (Couche 3)
============================================

QU'EST-CE QU'UN RÉGIME DE MARCHÉ ?
Un régime est la "personnalité" du marché à un instant donné.
Le même actif se comporte très différemment selon le régime.

LES 4 RÉGIMES DÉTECTÉS :
┌─────────────────┬──────────────────────────────────────────────────┐
│ Régime          │ Caractéristiques                                 │
├─────────────────┼──────────────────────────────────────────────────┤
│ 0 - BULL        │ Tendance haussière, faible volatilité            │
│                 │ → Favoriser les positions longues (achat)        │
├─────────────────┼──────────────────────────────────────────────────┤
│ 1 - BEAR        │ Tendance baissière, volatilité modérée-haute     │
│                 │ → Réduire l'exposition, positions courtes        │
├─────────────────┼──────────────────────────────────────────────────┤
│ 2 - RANGE       │ Marché sans direction, faible volatilité         │
│                 │ → Mean-reversion, vendre les extrêmes            │
├─────────────────┼──────────────────────────────────────────────────┤
│ 3 - HIGH VOL    │ Volatilité extrême, direction incertaine         │
│                 │ → Réduire les positions, stop-loss serrés        │
└─────────────────┴──────────────────────────────────────────────────┘

ALGORITHME : Hidden Markov Model (HMM)
Un HMM suppose que le marché est dans un état "caché" (le régime),
et que les observations (rendements, volatilité) dépendent de cet état.
Le modèle apprend automatiquement à détecter ces états cachés.

ANALOGIE :
Imaginez quelqu'un qui devine la météo (ensoleillé/pluvieux)
uniquement en observant si les gens portent un parapluie.
Le régime = météo cachée, les observations = indicateurs de marché.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from pathlib import Path
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from ..utils.logger import get_logger

logger = get_logger("RegimeDetector")

# Noms lisibles des régimes
REGIME_NAMES = {
    0: "BULL",
    1: "BEAR",
    2: "RANGE",
    3: "HIGH_VOL"
}


class MarketRegimeDetector:
    """
    Détecte le régime de marché courant via HMM.

    UTILISATION :
        detector = MarketRegimeDetector(config)
        detector.fit(df_features)
        regime, proba = detector.predict(df_features)
        print(f"Régime actuel : {REGIME_NAMES[regime[-1]]}")
    """

    def __init__(self, config: dict):
        self.config = config
        self.cfg = config["regime_detector"]
        self.n_regimes = self.cfg["n_regimes"]
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False

        # Pour fallback si hmmlearn non disponible
        self._use_kmeans_fallback = False

    def _build_regime_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Construit les features utilisées pour détecter les régimes.

        ON UTILISE UNIQUEMENT :
        - Rendement moyen sur 5 et 21 jours  → direction de la tendance
        - Volatilité sur 5 et 21 jours        → niveau de risque
        - Ratio volatilité court/long          → régime changeant ?
        - RSI 14                               → surachat/survente

        POURQUOI PEU DE FEATURES ICI ?
        Le HMM est un modèle probabiliste. Trop de features
        rendent l'estimation des paramètres instable.
        On garde les features les plus discriminantes.
        """
        features = pd.DataFrame(index=df.index)

        close = df["close"]

        # Rendements logarithmiques
        ret_1d  = np.log(close / close.shift(1)).fillna(0)
        ret_5d  = np.log(close / close.shift(5)).fillna(0)
        ret_21d = np.log(close / close.shift(21)).fillna(0)

        # Volatilité réalisée annualisée
        vol_5d  = ret_1d.rolling(5).std().fillna(0)  * np.sqrt(252)
        vol_21d = ret_1d.rolling(21).std().fillna(0) * np.sqrt(252)

        features["ret_5d"]       = ret_5d
        features["ret_21d"]      = ret_21d
        features["vol_5d"]       = vol_5d
        features["vol_21d"]      = vol_21d
        features["vol_ratio"]    = (vol_5d / (vol_21d + 1e-8)).clip(0, 5)

        # RSI si disponible
        if "rsi" in df.columns:
            features["rsi"] = df["rsi"].fillna(50) / 100.0  # Normaliser [0, 1]

        return features.fillna(0).values

    def fit(self, df: pd.DataFrame) -> "MarketRegimeDetector":
        """
        Entraîne le modèle HMM sur les données historiques.

        COMMENT LE HMM APPREND-IL ?
        L'algorithme Baum-Welch (variante de l'EM) :
        1. Initialise aléatoirement les paramètres du modèle
        2. E-step : calcule la probabilité de chaque régime pour chaque jour
        3. M-step : met à jour les paramètres pour maximiser la vraisemblance
        4. Répéter jusqu'à convergence

        PARAMÈTRE :
        - df : DataFrame avec colonnes OHLCV (+ features optionnelles)

        RETOURNE : self (pour chaînage)
        """
        logger.info(f"Entraînement du détecteur de régimes ({self.n_regimes} régimes)…")

        X_raw = self._build_regime_features(df)
        X = self.scaler.fit_transform(X_raw)

        # Essayer HMM d'abord, sinon KMeans comme fallback
        try:
            from hmmlearn.hmm import GaussianHMM

            self.model = GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="diag",      # Matrice de covariance diagonale
                n_iter=self.cfg["hmm_n_iter"],
                random_state=self.config["system"]["random_seed"],
                verbose=False
            )
            self.model.fit(X)
            self._use_kmeans_fallback = False
            logger.info("Modèle HMM (Gaussian) entraîné ✓")

        except ImportError:
            logger.warning("hmmlearn non disponible → fallback KMeans")
            self.model = KMeans(
                n_clusters=self.n_regimes,
                random_state=self.config["system"]["random_seed"],
                n_init=10
            )
            self.model.fit(X)
            self._use_kmeans_fallback = True
            logger.info("Modèle KMeans (fallback) entraîné ✓")

        self.is_fitted = True

        # Analyser les régimes appris
        regimes, _ = self.predict(df)
        self._analyze_regimes(df, regimes)

        return self

    def predict(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prédit le régime pour chaque jour.

        RETOURNE :
        - regimes  : Tableau d'entiers [0, 1, 2, 3] (un par jour)
        - probas   : Tableau de probabilités shape (n_jours, n_régimes)
                     probas[t] = [P(bull), P(bear), P(range), P(vol)]

        UTILISATION :
            regimes, probas = detector.predict(df)
            regime_aujourd_hui = regimes[-1]
            certitude = probas[-1].max() * 100
            print(f"Régime : {REGIME_NAMES[regime_aujourd_hui]} ({certitude:.0f}%)")
        """
        if not self.is_fitted:
            raise RuntimeError("Le modèle n'est pas encore entraîné. Appelez .fit() d'abord.")

        X_raw = self._build_regime_features(df)
        X = self.scaler.transform(X_raw)

        if self._use_kmeans_fallback:
            regimes = self.model.predict(X)
            # Créer des "probabilités" soft avec distances aux centroïdes
            distances = self.model.transform(X)
            inv_dist = 1.0 / (distances + 1e-8)
            probas = inv_dist / inv_dist.sum(axis=1, keepdims=True)
        else:
            regimes = self.model.predict(X)
            probas   = self.model.predict_proba(X)

        return regimes, probas

    def predict_current(
        self,
        df: pd.DataFrame
    ) -> Dict:
        """
        Prédit le régime ACTUEL (dernière observation).

        RETOURNE un dictionnaire avec :
        - regime_id    : Entier (0-3)
        - regime_name  : Nom lisible ("BULL", "BEAR"…)
        - probabilities: Probabilités de chaque régime
        - confidence   : Probabilité du régime dominant
        - strategy_hint: Conseil de stratégie

        EXEMPLE DE SORTIE :
        {
            "regime_id"    : 0,
            "regime_name"  : "BULL",
            "confidence"   : 0.87,
            "strategy_hint": "Favoriser les positions longues"
        }
        """
        regimes, probas = self.predict(df)
        current_regime = int(regimes[-1])
        current_probas = probas[-1]

        hints = {
            0: "Favoriser les positions longues (tendance haussière)",
            1: "Réduire l'exposition, envisager des positions courtes",
            2: "Stratégie mean-reversion, vendre les pics / acheter les creux",
            3: "ALERTE : Réduire toutes les positions, stop-loss serrés"
        }

        return {
            "regime_id"     : current_regime,
            "regime_name"   : REGIME_NAMES[current_regime],
            "probabilities" : {
                REGIME_NAMES[i]: float(p)
                for i, p in enumerate(current_probas)
            },
            "confidence"    : float(current_probas.max()),
            "strategy_hint" : hints[current_regime]
        }

    def _analyze_regimes(self, df: pd.DataFrame, regimes: np.ndarray):
        """
        Analyse et affiche les caractéristiques de chaque régime appris.
        Utile pour comprendre ce que le modèle a découvert.
        """
        close = df["close"]
        ret_1d = np.log(close / close.shift(1)).fillna(0)
        vol_21d = ret_1d.rolling(21).std().fillna(0) * np.sqrt(252)

        logger.info("─" * 55)
        logger.info("RÉGIMES APPRIS PAR LE MODÈLE :")
        logger.info(f"{'Régime':<15} {'% Temps':>8} {'Rendement/j':>12} {'Volatilité':>12}")
        logger.info("─" * 55)

        for r in range(self.n_regimes):
            mask = regimes == r
            if mask.sum() == 0:
                continue
            pct_time  = mask.mean() * 100
            avg_ret   = ret_1d[mask].mean() * 100
            avg_vol   = vol_21d[mask].mean() * 100
            logger.info(
                f"  Régime {r} ({REGIME_NAMES[r]:<8}) "
                f"{pct_time:>6.1f}%  "
                f"{avg_ret:>+10.3f}%  "
                f"{avg_vol:>10.1f}%"
            )
        logger.info("─" * 55)

    def save(self, path: str = "./models/regime_detector.pkl"):
        """Sauvegarde le modèle entraîné."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler,
                         "fallback": self._use_kmeans_fallback,
                         "fitted": self.is_fitted}, f)
        logger.info(f"Modèle sauvegardé : {path}")

    def load(self, path: str = "./models/regime_detector.pkl"):
        """Charge un modèle sauvegardé."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.model  = state["model"]
        self.scaler = state["scaler"]
        self._use_kmeans_fallback = state["fallback"]
        self.is_fitted = state["fitted"]
        logger.info(f"Modèle chargé depuis : {path}")
        return self
