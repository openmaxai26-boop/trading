"""
ORCHESTRATEUR D'INGÉNIERIE DES VARIABLES
==========================================

CE MODULE EST LE CHEF D'ORCHESTRE DES FEATURES.
Il appelle dans l'ordre :
  1. TechnicalFeatures   → RSI, MACD, Bollinger, VWAP, ATR
  2. StatisticalFeatures → Rendements, Volatilité, Sharpe, Skew
  3. MicrostructureFeatures → Order flow, Liquidité, Efficacité

RÉSULTAT FINAL :
Un DataFrame avec environ 60 colonnes de features prêtes
à être consommées par les modèles d'IA.

SÉLECTION DES FEATURES :
On applique une sélection pour éliminer les features
redondantes ou bruitées qui pourraient nuire aux modèles.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.impute import SimpleImputer

from .technical import TechnicalFeatures
from .statistical import StatisticalFeatures
from .microstructure import MicrostructureFeatures
from ..utils.logger import get_logger

logger = get_logger("FeatureEngineer")


class FeatureEngineer:
    """
    Orchestre la création de toutes les features pour un actif.

    UTILISATION :
        fe = FeatureEngineer(config)
        df_features = fe.engineer(df_ohlcv, symbol="AAPL")
        X, feature_names = fe.to_matrix(df_features)
    """

    def __init__(self, config: dict):
        self.config = config
        self.tech = TechnicalFeatures(config)
        self.stat = StatisticalFeatures(config)
        self.micro = MicrostructureFeatures(config)
        self.selected_features: Optional[List[str]] = None

    # ------------------------------------------------------------------ #
    #  COLONNES BRUTES (OHLCV) — à exclure de la matrice de features
    # ------------------------------------------------------------------ #
    _RAW_COLS = {"open", "high", "low", "close", "volume"}

    def engineer(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> pd.DataFrame:
        """
        Calcule TOUTES les features pour un actif donné.

        ÉTAPES :
        1. Indicateurs techniques (RSI, MACD, Bollinger…)
        2. Statistiques (volatilité, Sharpe, skew…)
        3. Microstructure (order flow, efficacité…)
        4. Imputation des valeurs manquantes résiduelles
        5. Remplacement des infinis

        PARAMÈTRES :
        - df     : DataFrame OHLCV brut
        - symbol : Nom de l'actif (pour les logs)

        RETOURNE :
        - DataFrame enrichi de ~60 colonnes
        """
        logger.info(f"Ingénierie des features pour {symbol}…")

        # --- 1. Indicateurs techniques ---
        df = self.tech.compute_all(df)

        # --- 2. Statistiques ---
        df = self.stat.compute_all(df)

        # --- 3. Microstructure ---
        df = self.micro.compute_all(df)

        # --- 4. Dédupliquer les colonnes (au cas où concat crée des doublons) ---
        df = df.loc[:, ~df.columns.duplicated()]

        # --- 5. Nettoyer les valeurs infinies ---
        df = df.replace([np.inf, -np.inf], np.nan)

        # --- 6. Imputation des NaN résiduels (forward fill puis médiane) ---
        df = df.ffill().bfill()
        # Pour les colonnes encore NaN (début de série), utiliser 0
        nan_mask = df.isnull().any(axis=0)
        for col in df.columns[nan_mask]:
            median_val = df[col].median()
            fill_val = float(median_val) if not np.isnan(float(median_val)) else 0.0
            df[col] = df[col].fillna(fill_val)

        # --- 6. Supprimer les premières lignes incomplètes ---
        # Les indicateurs ont besoin d'un "warmup" (ex: EMA 200 → 200 premières lignes NaN)
        warmup = 210   # 200 jours pour EMA 200 + marge de sécurité
        if len(df) > warmup:
            df = df.iloc[warmup:].copy()

        n_features = len([c for c in df.columns if c not in self._RAW_COLS])
        logger.info(f"  {symbol} : {n_features} features calculées sur {len(df)} lignes ✓")

        return df

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Retourne la liste des colonnes de features (hors OHLCV bruts).

        POURQUOI EXCLURE OHLCV ?
        Les prix bruts ne sont pas normalisés de façon comparable
        entre actifs. On utilise les features dérivées (rendements,
        indicateurs normalisés) pour l'apprentissage.
        """
        return [c for c in df.columns if c not in self._RAW_COLS]

    def to_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Convertit le DataFrame enrichi en matrice numpy pour l'IA.

        RETOURNE :
        - X             : Matrice (n_jours × n_features)
        - feature_names : Noms des colonnes (pour l'interprétabilité)
        """
        feat_cols = self.selected_features or self.get_feature_columns(df)
        # Garder seulement les colonnes présentes
        feat_cols = [c for c in feat_cols if c in df.columns]
        X = df[feat_cols].values.astype(np.float32)
        return X, feat_cols

    def select_features(
        self,
        df: pd.DataFrame,
        target: np.ndarray,
        k: int = 40
    ) -> List[str]:
        """
        Sélectionne les K meilleures features par information mutuelle.

        QU'EST-CE QUE L'INFORMATION MUTUELLE ?
        C'est une mesure de la dépendance entre deux variables.
        Elle capture les relations NON-LINÉAIRES (contrairement
        à la corrélation de Pearson).

        POURQUOI SÉLECTIONNER LES FEATURES ?
        Trop de features = "curse of dimensionality" (malédiction
        de la dimensionnalité). Un modèle avec 100 features
        redondantes apprend du bruit, pas du signal.

        RÈGLE PRATIQUE :
        Garder les 30-50 features les plus informatives.

        PARAMÈTRES :
        - df     : DataFrame avec toutes les features
        - target : Variable cible (rendements futurs)
        - k      : Nombre de features à conserver

        RETOURNE :
        - Liste des noms des meilleures features
        """
        feat_cols = self.get_feature_columns(df)
        X = df[feat_cols].values
        y = target[:len(X)]

        # Imputer les NaN pour le calcul (mutual_info ne gère pas les NaN)
        imputer = SimpleImputer(strategy="median")
        X_imputed = imputer.fit_transform(X)

        k = min(k, X_imputed.shape[1])

        selector = SelectKBest(score_func=mutual_info_regression, k=k)
        selector.fit(X_imputed, y)

        selected_mask = selector.get_support()
        selected = [col for col, keep in zip(feat_cols, selected_mask) if keep]

        logger.info(f"Sélection de features : {len(feat_cols)} → {len(selected)} features ✓")

        # Afficher le top 10
        scores = dict(zip(feat_cols, selector.scores_))
        top10 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info("Top 10 features par importance :")
        for name, score in top10:
            logger.info(f"  {name:35s} : {score:.4f}")

        self.selected_features = selected
        return selected

    def engineer_all(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Applique l'ingénierie des features à tous les actifs.

        PARAMÈTRE :
        - data : Dictionnaire {symbole: DataFrame OHLCV}

        RETOURNE :
        - Dictionnaire {symbole: DataFrame enrichi}
        """
        result = {}
        for symbol, df in data.items():
            try:
                result[symbol] = self.engineer(df, symbol)
            except Exception as e:
                logger.error(f"Erreur features pour {symbol} : {e}")
        return result
