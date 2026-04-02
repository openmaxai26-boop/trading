"""
PIPELINE DE DONNÉES COMPLET
============================

QU'EST-CE QU'UN PIPELINE ?
Un pipeline est une chaîne d'opérations séquentielles.
Comme une chaîne de montage dans une usine :
  Données brutes → Nettoyage → Features → Séquences → Prêt pour l'IA

CE MODULE ORCHESTRE :
1. La collecte (DataCollector)
2. Le prétraitement (DataPreprocessor)
3. La création de séquences temporelles

QU'EST-CE QU'UNE SÉQUENCE TEMPORELLE ?
Les modèles LSTM et Transformer ont besoin de SÉQUENCES.
Au lieu d'une seule ligne de données (un jour),
on leur donne les 60 derniers jours pour prédire le suivant.

EXEMPLE :
Fenêtre d'entrée : [Jour 1, Jour 2, ..., Jour 60]
                              ↓
                    MODÈLE (LSTM/Transformer)
                              ↓
            Prédiction : probabilité de hausse Jour 61
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader

from .collectors import MarketDataCollector
from .preprocessor import DataPreprocessor
from ..utils.logger import get_logger

logger = get_logger("DataPipeline")


class TimeSeriesDataset(Dataset):
    """
    Dataset PyTorch pour données de séries temporelles.

    QU'EST-CE QU'UN DATASET PYTORCH ?
    PyTorch (le framework de deep learning) a besoin que les données
    soient encapsulées dans un objet Dataset pour pouvoir les
    charger efficacement en mini-lots (batches) pendant l'entraînement.

    FONCTIONNEMENT :
    Avec une fenêtre de 60 jours et 500 jours de données :
    - Séquence 1 : jours 0-59  → cible : jour 60
    - Séquence 2 : jours 1-60  → cible : jour 61
    - Séquence 3 : jours 2-61  → cible : jour 62
    - ...
    Total : 440 séquences d'entraînement

    PARAMÈTRES :
    - features    : Tableau numpy des features (jours × features)
    - targets     : Tableau numpy des cibles (rendements futurs)
    - sequence_len: Nombre de jours dans chaque fenêtre (ex: 60)
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        sequence_len: int = 60
    ):
        self.sequence_len = sequence_len

        # Vérification des dimensions
        assert len(features) == len(targets), \
            f"Dimensions incompatibles : features={len(features)}, targets={len(targets)}"

        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self) -> int:
        """Retourne le nombre total de séquences disponibles."""
        return len(self.features) - self.sequence_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retourne une séquence (fenêtre) et sa cible.

        PARAMÈTRES :
        - idx : Index de la séquence (0 à len-1)

        RETOURNE :
        - x : Tenseur de forme (sequence_len, n_features)
        - y : Tenseur scalaire (rendement futur)
        """
        x = self.features[idx : idx + self.sequence_len]
        y = self.targets[idx + self.sequence_len]
        return x, y


class DataPipeline:
    """
    Pipeline complet : de la collecte à la préparation pour l'IA.

    UTILISATION TYPIQUE :
        pipeline = DataPipeline(config)
        train_loader, val_loader, test_loader = pipeline.run("AAPL")

    SÉPARATION TRAIN/VAL/TEST :
    - TRAIN (70%) : Le modèle APPREND sur ces données
    - VALIDATION (15%) : On AJUSTE les hyperparamètres
    - TEST (15%) : On ÉVALUE les performances finales (jamais vu)

    POURQUOI CETTE SÉPARATION ?
    Imaginez un examen scolaire :
    - Train = exercices d'entraînement
    - Validation = devoirs maison (ajuster sa méthode)
    - Test = examen final (vraie évaluation)
    Si on entraîne sur les données de test → le modèle "triche" !
    """

    def __init__(self, config: dict):
        self.config = config
        self.collector = MarketDataCollector(config)
        self.preprocessor = DataPreprocessor(config)
        self.sequence_len = config["features"]["sequence_length"]
        self.batch_size = config["prediction_model"]["training"]["batch_size"]

        # Ratios de séparation depuis la configuration
        self.train_ratio = config["timeframes"]["train_ratio"]
        self.val_ratio = config["timeframes"]["val_ratio"]

    def compute_targets(self, df: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        """
        Calcule les variables cibles (ce qu'on veut prédire).

        CIBLE = Rendement futur sur 'horizon' jours
        Rendement = (Prix demain - Prix aujourd'hui) / Prix aujourd'hui

        EXEMPLE :
        - Prix aujourd'hui : 100$
        - Prix demain : 102$
        - Rendement = (102-100)/100 = +2%

        Pour une classification (HAUSSE/BAISSE), on utilise :
        - 1 si rendement > 0 (prix monte)
        - 0 si rendement ≤ 0 (prix baisse ou stable)

        PARAMÈTRES :
        - df      : DataFrame avec colonne 'close'
        - horizon : Nombre de jours à l'avance à prédire

        RETOURNE :
        - Tableau numpy des rendements futurs
        """
        # Rendement logarithmique (plus stable mathématiquement)
        log_returns = np.log(df["close"] / df["close"].shift(horizon))
        return log_returns.fillna(0).values

    def split_data(
        self,
        features: np.ndarray,
        targets: np.ndarray
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray]
    ]:
        """
        Divise les données en train/validation/test.

        IMPORTANT : On divise TEMPORELLEMENT (chronologiquement).
        Ne jamais mélanger aléatoirement les données de séries temporelles !

        POURQUOI ?
        Si on mélange : une séquence de test pourrait contenir des
        informations du futur (dans les données d'entraînement).
        C'est du "look-ahead bias" → les performances seraient fausses.

        Données temporelles : [====Train====|==Val==|=Test=]
                              (70%)          (15%)   (15%)
        """
        n = len(features)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        train = (features[:train_end], targets[:train_end])
        val = (features[train_end:val_end], targets[train_end:val_end])
        test = (features[val_end:], targets[val_end:])

        logger.info(
            f"Séparation des données : "
            f"Train={len(train[0])} | Val={len(val[0])} | Test={len(test[0])}"
        )

        return train, val, test

    def create_dataloader(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        shuffle: bool = False
    ) -> DataLoader:
        """
        Crée un DataLoader PyTorch à partir des données.

        QU'EST-CE QU'UN DATALOADER ?
        Un DataLoader divise automatiquement les données en mini-lots
        (batches) et les fournit au modèle pendant l'entraînement.

        EXEMPLE avec batch_size=64 :
        Si on a 1000 séquences, le DataLoader crée 15 mini-lots de 64
        et 1 mini-lot de 40.

        POURQUOI shuffle=False pour test/val ?
        On veut que les données de validation et test restent
        dans l'ordre chronologique pour l'évaluation.
        """
        dataset = TimeSeriesDataset(features, targets, self.sequence_len)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,      # 0 = pas de parallélisme (plus stable)
            pin_memory=False    # True = accélère si GPU disponible
        )

    def run(
        self,
        symbol: Optional[str] = None,
        raw_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Dict[str, Dict[str, DataLoader]]:
        """
        Exécute le pipeline complet pour un ou tous les actifs.

        PARAMÈTRES :
        - symbol   : Si fourni, traiter seulement cet actif
        - raw_data : Si fourni, utiliser ces données (sinon télécharger)

        RETOURNE :
        Dictionnaire {symbole: {"train": DataLoader, "val": DataLoader, "test": DataLoader}}

        UTILISATION :
            # Pour tous les actifs
            loaders = pipeline.run()

            # Pour un seul actif
            loaders = pipeline.run("AAPL")

            # Entraînement
            for X, y in loaders["AAPL"]["train"]:
                # X: (batch_size, sequence_len, n_features)
                # y: (batch_size,)
                ...
        """
        # Étape 1 : Collecter les données si non fournies
        if raw_data is None:
            logger.info("Collecte des données depuis les APIs...")
            raw_data = self.collector.get_all_data()

        # Étape 2 : Prétraiter
        logger.info("Prétraitement des données...")
        processed_data = self.preprocessor.process(raw_data, fit=True)

        # Filtrer si un seul symbole demandé
        if symbol is not None:
            if symbol not in processed_data:
                raise ValueError(
                    f"Symbole '{symbol}' non trouvé. "
                    f"Disponibles : {list(processed_data.keys())}"
                )
            processed_data = {symbol: processed_data[symbol]}

        # Étape 3 : Créer les features et séquences
        all_loaders = {}

        for sym, df in processed_data.items():
            logger.info(f"Création des séquences pour {sym}...")

            try:
                # Les features = les colonnes OHLCV normalisées
                # (les indicateurs techniques seront ajoutés par FeatureEngineer)
                features = df[["open", "high", "low", "close", "volume"]].values
                targets = self.compute_targets(df)

                # Vérification
                if len(features) < self.sequence_len * 2:
                    logger.warning(
                        f"  {sym} : trop peu de données "
                        f"({len(features)} < {self.sequence_len * 2}) → ignoré"
                    )
                    continue

                # Séparation train/val/test
                (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
                    self.split_data(features, targets)

                # Création des DataLoaders
                all_loaders[sym] = {
                    "train": self.create_dataloader(X_train, y_train, shuffle=True),
                    "val": self.create_dataloader(X_val, y_val, shuffle=False),
                    "test": self.create_dataloader(X_test, y_test, shuffle=False),
                    "raw_df": df,       # Garder le DataFrame brut pour le backtesting
                    "n_features": features.shape[1]
                }

                logger.info(
                    f"  {sym} : "
                    f"train={len(X_train)} | val={len(X_val)} | test={len(X_test)} ✓"
                )

            except Exception as e:
                logger.error(f"  {sym} : Erreur lors de la création du pipeline : {e}")
                import traceback
                logger.debug(traceback.format_exc())

        logger.info(f"Pipeline terminé : {len(all_loaders)} actifs prêts")
        return all_loaders
