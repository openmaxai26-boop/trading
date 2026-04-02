"""
PRÉPROCESSEUR DE DONNÉES
=========================

QU'EST-CE QUE LE PRÉTRAITEMENT ?
Les données brutes de marché contiennent souvent des problèmes :
- Valeurs manquantes (jours fériés, problèmes de collecte)
- Valeurs aberrantes (erreurs de données, flash crashes)
- Échelles différentes (AAPL à 180$, BTC à 40,000$)

Ce module nettoie et normalise les données pour les rendre
utilisables par les modèles d'IA.

ÉTAPES DU PRÉTRAITEMENT :
1. Suppression des valeurs manquantes (ou remplissage)
2. Détection et traitement des valeurs aberrantes
3. Normalisation (mise à l'échelle similaire)
4. Alignement temporel (même dates pour tous les actifs)

POURQUOI NORMALISER ?
Un réseau de neurones compare des chiffres.
Si AAPL = 180 et BTC = 40,000, le modèle pensera que BTC est
200x plus important. La normalisation met tout entre -1 et 1
(ou 0 et 1) pour traiter équitablement tous les actifs.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pickle
from pathlib import Path

from ..utils.logger import get_logger

logger = get_logger("DataPreprocessor")


class DataPreprocessor:
    """
    Nettoie et normalise les données de marché brutes.

    COMMENT L'UTILISER :
        preprocessor = DataPreprocessor(config)
        clean_data = preprocessor.process(raw_data)
    """

    def __init__(self, config: dict):
        self.config = config
        self.cleaning_config = config["data"]["cleaning"]
        self.norm_method = config["features"]["normalization"]

        # Dictionnaire pour stocker les scalers (un par actif)
        # Nécessaire pour inverser la normalisation lors des prédictions
        self.scalers: Dict[str, object] = {}

        self.processed_dir = Path(config["data"]["processed_dir"])
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def remove_outliers(self, df: pd.DataFrame, threshold: float = 5.0) -> pd.DataFrame:
        """
        Supprime les valeurs aberrantes basées sur le z-score.

        QU'EST-CE QU'UN Z-SCORE ?
        Le z-score mesure à combien d'écarts-types une valeur est
        éloignée de la moyenne.
        - z = 0 → valeur égale à la moyenne
        - z = 2 → valeur 2 fois l'écart-type au-dessus
        - z > 5 → presque certainement une erreur de données

        EXEMPLE :
        Si le prix moyen de BTC est 40,000$ et l'écart-type 5,000$,
        une valeur de 100,000$ aurait z = (100,000-40,000)/5,000 = 12
        → c'est une aberration à corriger.

        PARAMÈTRES :
        - df        : DataFrame OHLCV
        - threshold : Seuil z-score (5.0 = conserver 99.99% des données)
        """
        df_clean = df.copy()

        for col in ["open", "high", "low", "close"]:
            if col not in df_clean.columns:
                continue
            # Calcul des rendements logarithmiques (plus stable que les prix bruts)
            returns = df_clean[col].pct_change().dropna()
            z_scores = np.abs((returns - returns.mean()) / (returns.std() + 1e-8))
            outlier_mask = z_scores > threshold

            if outlier_mask.sum() > 0:
                logger.warning(
                    f"  {outlier_mask.sum()} valeurs aberrantes détectées dans '{col}' → remplacées"
                )
                # Remplacer par la valeur précédente (forward fill)
                df_clean.loc[outlier_mask[outlier_mask].index, col] = np.nan
                df_clean[col] = df_clean[col].ffill()

        return df_clean

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Gère les valeurs manquantes dans les données.

        STRATÉGIES DISPONIBLES :
        - "forward" : Utiliser la dernière valeur connue (méthode standard)
        - "interpolate" : Interpolation linéaire entre les valeurs connues
        - "drop" : Supprimer les lignes avec valeurs manquantes

        POURQUOI "FORWARD FILL" POUR LES PRIX ?
        Si une bourse est fermée un lundi (jour férié), le prix du
        lundi est "inconnu" mais on peut supposer qu'il est le même
        que le vendredi précédent. C'est la convention standard.
        """
        method = self.cleaning_config.get("fill_missing", "forward")
        missing_before = df.isnull().sum().sum()

        if missing_before > 0:
            if method == "forward":
                df = df.ffill().bfill()
            elif method == "interpolate":
                df = df.interpolate(method="linear").bfill()
            elif method == "drop":
                df = df.dropna()

            missing_after = df.isnull().sum().sum()
            logger.debug(
                f"  Valeurs manquantes : {missing_before} → {missing_after} après remplissage"
            )

        return df

    def normalize(
        self,
        df: pd.DataFrame,
        symbol: str,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Normalise les données pour les modèles de machine learning.

        MÉTHODES DE NORMALISATION :
        1. "zscore" (StandardScaler) : moyenne=0, écart-type=1
           → Idéal pour LSTM et Transformers
           Formule : z = (x - moyenne) / écart-type

        2. "minmax" (MinMaxScaler) : valeurs entre 0 et 1
           → Idéal pour les CNN
           Formule : z = (x - min) / (max - min)

        3. "robust" (RobustScaler) : robuste aux outliers
           → Utilise la médiane et les quartiles
           Formule : z = (x - médiane) / IQR

        IMPORTANT : On "fit" (apprend les paramètres) uniquement sur
        les données d'ENTRAÎNEMENT, jamais sur les données de test !
        C'est pour éviter le "data leakage" (fuite d'information).

        PARAMÈTRES :
        - df     : DataFrame à normaliser
        - symbol : Nom de l'actif (pour stocker le scaler)
        - fit    : True = apprendre les paramètres | False = utiliser scaler existant
        """
        method = self.norm_method

        if method == "zscore":
            ScalerClass = StandardScaler
        elif method == "minmax":
            ScalerClass = MinMaxScaler
        elif method == "robust":
            ScalerClass = RobustScaler
        else:
            logger.warning(f"Méthode de normalisation '{method}' inconnue → zscore par défaut")
            ScalerClass = StandardScaler

        df_norm = df.copy()
        cols_to_normalize = [c for c in df.columns if c != "volume"]

        # Normalisation du volume séparément (distribution très différente)
        volume_cols = [c for c in df.columns if c == "volume"]

        if fit:
            self.scalers[symbol] = {}

            # Scaler pour les prix
            if cols_to_normalize:
                scaler_price = ScalerClass()
                df_norm[cols_to_normalize] = scaler_price.fit_transform(
                    df[cols_to_normalize]
                )
                self.scalers[symbol]["price"] = scaler_price

            # Scaler pour le volume (log-transformé d'abord)
            if volume_cols:
                scaler_vol = ScalerClass()
                log_vol = np.log1p(df[volume_cols])  # log(1+x) pour stabiliser
                df_norm[volume_cols] = scaler_vol.fit_transform(log_vol)
                self.scalers[symbol]["volume"] = scaler_vol
        else:
            # Utiliser les scalers déjà appris
            if symbol not in self.scalers:
                logger.error(f"Aucun scaler trouvé pour {symbol} → fit d'abord !")
                return df_norm

            if cols_to_normalize and "price" in self.scalers[symbol]:
                df_norm[cols_to_normalize] = self.scalers[symbol]["price"].transform(
                    df[cols_to_normalize]
                )

            if volume_cols and "volume" in self.scalers[symbol]:
                log_vol = np.log1p(df[volume_cols])
                df_norm[volume_cols] = self.scalers[symbol]["volume"].transform(log_vol)

        return df_norm

    def denormalize_price(self, normalized_price: np.ndarray, symbol: str) -> np.ndarray:
        """
        Convertit un prix normalisé en prix réel.

        POURQUOI EN A-T-ON BESOIN ?
        Le modèle prédit des valeurs normalisées (ex: 0.7).
        Pour afficher "AAPL sera à environ 185$", on doit
        inverser la normalisation.

        PARAMÈTRES :
        - normalized_price : Valeur(s) normalisée(s)
        - symbol           : Nom de l'actif

        RETOURNE : Prix en dollars (ou devise de l'actif)
        """
        if symbol not in self.scalers or "price" not in self.scalers[symbol]:
            logger.warning(f"Scaler non trouvé pour {symbol}")
            return normalized_price

        scaler = self.scalers[symbol]["price"]
        # Le scaler attend un tableau 2D
        price_reshaped = normalized_price.reshape(-1, 1)
        return scaler.inverse_transform(
            np.hstack([price_reshaped] + [np.zeros_like(price_reshaped)] * (scaler.n_features_in_ - 1))
        )[:, 0]

    def align_dates(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Aligne tous les actifs sur les mêmes dates.

        PROBLÈME :
        - AAPL (action) a des données du lundi au vendredi
        - BTC-USD (crypto) a des données 7j/7
        - GC=F (or) a ses propres jours de marché

        SOLUTION :
        Créer un calendrier commun = union de toutes les dates,
        puis remplir les trous avec la dernière valeur connue.

        PARAMÈTRES :
        - data : Dictionnaire {symbole: DataFrame}

        RETOURNE : Dictionnaire avec dates alignées
        """
        if not data:
            return data

        # Trouver la date de début et de fin communes
        start_dates = [df.index.min() for df in data.values() if not df.empty]
        end_dates = [df.index.max() for df in data.values() if not df.empty]

        if not start_dates:
            return data

        common_start = max(start_dates)
        common_end = min(end_dates)

        logger.info(f"Alignement des dates : {common_start.date()} → {common_end.date()}")

        aligned = {}
        for symbol, df in data.items():
            if df.empty:
                continue
            # Filtrer sur la plage commune
            df_aligned = df.loc[common_start:common_end].copy()
            df_aligned = df_aligned.ffill().bfill()
            aligned[symbol] = df_aligned

        return aligned

    def save_scalers(self, path: str = "./data/scalers.pkl"):
        """Sauvegarde les scalers pour utilisation future (production)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.scalers, f)
        logger.info(f"Scalers sauvegardés : {path}")

    def load_scalers(self, path: str = "./data/scalers.pkl"):
        """Charge les scalers sauvegardés."""
        with open(path, "rb") as f:
            self.scalers = pickle.load(f)
        logger.info(f"Scalers chargés depuis : {path}")

    def process(
        self,
        raw_data: Dict[str, pd.DataFrame],
        fit: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Pipeline complet de prétraitement : nettoie + normalise toutes les données.

        ÉTAPES :
        1. Vérification de la qualité des données
        2. Suppression des valeurs aberrantes
        3. Remplissage des valeurs manquantes
        4. Alignement des dates
        5. Normalisation

        PARAMÈTRES :
        - raw_data : Dictionnaire {symbole: DataFrame brut}
        - fit      : True = apprendre les paramètres de normalisation

        RETOURNE :
        - Dictionnaire {symbole: DataFrame nettoyé et normalisé}
        """
        logger.info("Démarrage du prétraitement des données...")
        processed = {}
        min_history = self.cleaning_config.get("min_history_days", 365)

        for symbol, df in raw_data.items():
            logger.info(f"Prétraitement de {symbol}...")

            if df is None or df.empty:
                logger.warning(f"  {symbol} : données vides → ignoré")
                continue

            # Vérification de l'historique minimum
            if len(df) < min_history:
                logger.warning(
                    f"  {symbol} : seulement {len(df)} jours "
                    f"(minimum requis : {min_history}) → ignoré"
                )
                continue

            # Étape 1 : Supprimer les outliers
            if self.cleaning_config.get("remove_outliers", True):
                df = self.remove_outliers(
                    df, self.cleaning_config.get("outlier_threshold", 5.0)
                )

            # Étape 2 : Gérer les valeurs manquantes
            df = self.handle_missing_values(df)

            # Étape 3 : S'assurer que les colonnes sont du bon type
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna()

            processed[symbol] = df
            logger.info(f"  {symbol} : prétraitement terminé ({len(df)} lignes) ✓")

        # Étape 4 : Aligner les dates
        processed = self.align_dates(processed)

        logger.info(f"Prétraitement terminé : {len(processed)} actifs traités")
        return processed
