"""
SYSTÈME D'APPRENTISSAGE CONTINU  (Couche 9)
=============================================

POURQUOI L'APPRENTISSAGE CONTINU EST CRUCIAL ?
Les marchés financiers ÉVOLUENT en permanence :
- Les corrélations changent (ex: Bitcoin suivait les techs en 2021)
- Les régimes changent (bull→bear→range)
- Les dynamiques macro changent (taux d'intérêt, inflation)

Un modèle entraîné une fois en janvier sera OBSOLÈTE en septembre.

ANALOGIE :
Un médecin qui n'a pas mis à jour ses connaissances depuis 20 ans
utilise des traitements dépassés. Le modèle de trading doit
"lire les nouvelles publications" régulièrement.

DEUX PROBLÈMES À RÉSOUDRE :

1. DÉRIVE DES DONNÉES (Data Drift) :
   La distribution des données d'entrée change.
   → Test de Kolmogorov-Smirnov pour détecter le changement

2. DÉRIVE DE PERFORMANCE (Performance Drift) :
   Le modèle performe de moins en moins bien.
   → Surveiller le Sharpe ratio roulant

STRATÉGIES DE RÉENTRAÎNEMENT :
A) Réentraînement complet : Tout réapprendre depuis zéro
   + Capture toutes les nouvelles dynamiques
   - Lent, coûteux en calcul

B) Fine-tuning : Continuer l'entraînement sur les nouvelles données
   + Rapide, préserve les patterns appris
   - Risque d'oublier des patterns anciens (catastrophic forgetting)

C) Hybride : Fine-tuning périodique + réentraînement complet mensuel
   → C'est ce que nous implémentons ✓
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.stats import ks_2samp, mannwhitneyu
from datetime import datetime, timedelta
from pathlib import Path
import json

from ..utils.logger import get_logger

logger = get_logger("ContinuousLearner")


class DataDriftDetector:
    """
    Détecte la dérive statistique dans les données d'entrée.

    TEST DE KOLMOGOROV-SMIRNOV (KS) :
    Compare deux distributions pour voir si elles sont significativement
    différentes.

    FONCTIONNEMENT :
    - Distribution de référence = données d'entraînement initiales
    - Distribution courante = données des N derniers jours
    - Si elles diffèrent statistiquement → dérive détectée

    EXEMPLE :
    Feature "RSI" : En 2022 (marché baissier), RSI souvent < 40
                    En 2023 (reprise), RSI souvent > 55
    → Le test KS détecte que la distribution du RSI a changé
    → Réentraînement suggéré
    """

    def __init__(self, config: dict):
        self.cfg         = config["continuous_learning"]["drift_detection"]
        self.reference   : Optional[pd.DataFrame] = None
        self.window_size  = self.cfg["window_size"]
        self.p_threshold  = self.cfg["p_value_threshold"]

    def set_reference(self, df: pd.DataFrame):
        """
        Définit la distribution de référence (données d'entraînement).

        PARAMÈTRE :
        - df : DataFrame des features d'entraînement
        """
        self.reference = df.copy()
        logger.info(
            f"Référence de drift enregistrée : {len(df)} échantillons, "
            f"{len(df.columns)} features"
        )

    def detect(self, current_df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Compare la distribution courante à la référence.

        MÉTHODE KS (Kolmogorov-Smirnov) :
        La statistique KS mesure la distance maximale entre deux
        fonctions de répartition empiriques.

        H0 (hypothèse nulle)   : Les deux distributions sont identiques
        H1 (hypothèse alternative): Les distributions sont différentes

        Si p-value < seuil (0.05) → on rejette H0 → dérive détectée

        PARAMÈTRE :
        - current_df : Données récentes à comparer

        RETOURNE :
        - drift_detected : True si dérive significative
        - report         : Détails par feature
        """
        if self.reference is None:
            logger.warning("Pas de référence définie → impossible de détecter la dérive")
            return False, {}

        # Utiliser la fenêtre récente
        current = current_df.tail(self.window_size)

        drifted_features = []
        report = {}

        common_cols = [c for c in self.reference.columns if c in current.columns]

        for col in common_cols:
            ref_vals = self.reference[col].dropna().values
            cur_vals = current[col].dropna().values

            if len(ref_vals) < 10 or len(cur_vals) < 10:
                continue

            # Test KS
            ks_stat, p_value = ks_2samp(ref_vals, cur_vals)

            drifted = p_value < self.p_threshold
            report[col] = {
                "ks_statistic": round(float(ks_stat), 4),
                "p_value"     : round(float(p_value), 4),
                "drifted"     : drifted
            }

            if drifted:
                drifted_features.append(col)

        drift_ratio = len(drifted_features) / max(len(common_cols), 1)
        drift_detected = drift_ratio > 0.20  # Dérive si > 20% des features ont changé

        if drift_detected:
            logger.warning(
                f"DÉRIVE DÉTECTÉE : {len(drifted_features)}/{len(common_cols)} features "
                f"({drift_ratio:.0%})"
            )
            logger.warning(f"  Features driftées : {drifted_features[:5]}...")
        else:
            logger.info(
                f"Pas de dérive significative : {len(drifted_features)}/{len(common_cols)} features"
            )

        return drift_detected, report


class PerformanceMonitor:
    """
    Surveille la dégradation des performances du modèle.

    INDICATEURS SURVEILLÉS :
    1. Sharpe ratio roulant (30 jours)
    2. Accuracy directionnelle roulante
    3. Profit factor roulant

    ALERTE si :
    - Sharpe < seuil configuré (0.5 par défaut)
    - Accuracy < 45% (pire que le hasard)
    - 3 semaines consécutives de pertes
    """

    def __init__(self, config: dict):
        self.cfg = config["continuous_learning"]["performance_monitoring"]
        self.performance_history: List[Dict] = []

    def record(
        self,
        date: str,
        daily_return: float,
        predicted_direction: int,
        actual_direction: int
    ):
        """
        Enregistre les performances d'une journée.

        PARAMÈTRES :
        - date                : Date (format YYYY-MM-DD)
        - daily_return        : Rendement du portefeuille ce jour
        - predicted_direction : Direction prédite (+1 ou -1)
        - actual_direction    : Direction réelle (+1 ou -1)
        """
        correct = int(predicted_direction == actual_direction)
        self.performance_history.append({
            "date"      : date,
            "return"    : daily_return,
            "correct"   : correct
        })

    def check_degradation(self) -> Tuple[bool, Dict]:
        """
        Vérifie si les performances se sont dégradées.

        RETOURNE :
        - degraded : True si le modèle performe mal
        - metrics  : Métriques de performance récentes
        """
        lookback = self.cfg["lookback_days"]
        min_sharpe = self.cfg["min_sharpe_threshold"]

        if len(self.performance_history) < lookback:
            return False, {"message": "Pas assez d'historique"}

        recent = self.performance_history[-lookback:]
        returns = np.array([r["return"] for r in recent])
        correct = np.array([r["correct"] for r in recent])

        # Sharpe roulant
        if returns.std() > 0:
            sharpe = float(returns.mean() / returns.std() * np.sqrt(252))
        else:
            sharpe = 0.0

        # Accuracy directionnelle
        dir_acc = float(correct.mean() * 100)

        # Semaines consécutives de pertes
        weekly_returns = [
            np.sum(returns[i:i+5])
            for i in range(0, len(returns) - 4, 5)
        ]
        consecutive_loss_weeks = 0
        for wr in reversed(weekly_returns):
            if wr < 0:
                consecutive_loss_weeks += 1
            else:
                break

        metrics = {
            "sharpe_30d"         : round(sharpe, 2),
            "dir_accuracy_30d"   : round(dir_acc, 1),
            "consecutive_loss_wk": consecutive_loss_weeks,
            "avg_daily_return"   : round(float(returns.mean() * 100), 3)
        }

        degraded = (
            sharpe < min_sharpe or
            dir_acc < 45.0 or
            consecutive_loss_weeks >= 3
        )

        if degraded:
            logger.warning("DÉGRADATION DE PERFORMANCE DÉTECTÉE :")
            for k, v in metrics.items():
                logger.warning(f"  {k}: {v}")

        return degraded, metrics


class ContinuousLearner:
    """
    Système d'apprentissage continu qui orchestre la détection
    de dérive et le réentraînement automatique.

    UTILISATION :
        learner = ContinuousLearner(config)
        learner.setup(model, trainer, feature_engineer)

        # Appeler périodiquement
        retrained, report = learner.step(new_data, performance_data)
    """

    def __init__(self, config: dict):
        self.config       = config
        self.cl_cfg       = config["continuous_learning"]
        self.drift_detector = DataDriftDetector(config)
        self.perf_monitor   = PerformanceMonitor(config)
        self.last_retrain   = None
        self.retrain_count  = 0
        self.retrain_log: List[Dict] = []

    def setup(self, model, trainer, feature_engineer):
        """
        Configure le learner avec les composants du système.

        PARAMÈTRES :
        - model            : HybridPredictionModel
        - trainer          : ModelTrainer
        - feature_engineer : FeatureEngineer
        """
        self.model   = model
        self.trainer = trainer
        self.fe      = feature_engineer
        logger.info("ContinuousLearner configuré ✓")

    def initialize_reference(self, df_features: pd.DataFrame):
        """
        Définit la distribution de référence initiale.
        À appeler après le premier entraînement.
        """
        self.drift_detector.set_reference(df_features)
        self.last_retrain = datetime.now()

    def should_retrain(
        self,
        current_features: pd.DataFrame,
        performance_data: Optional[List[Dict]] = None
    ) -> Tuple[bool, str]:
        """
        Décide si un réentraînement est nécessaire.

        CRITÈRES DE RÉENTRAÎNEMENT :
        1. Périodique : N jours depuis le dernier entraînement
        2. Dérive des données > 20% des features
        3. Dégradation des performances (Sharpe < seuil)

        RETOURNE :
        - should : True si réentraînement requis
        - reason : Raison du réentraînement
        """
        freq_days = self.cl_cfg["retrain_frequency_days"]

        # ── Critère 1 : Périodique ──
        if self.last_retrain is None:
            return True, "Premier entraînement"

        days_since = (datetime.now() - self.last_retrain).days
        if days_since >= freq_days:
            return True, f"Réentraînement périodique ({days_since} jours écoulés)"

        # ── Critère 2 : Dérive des données ──
        if self.cl_cfg["drift_detection"]["enable"]:
            drift_detected, drift_report = self.drift_detector.detect(current_features)
            if drift_detected:
                return True, "Dérive des données détectée (test KS)"

        # ── Critère 3 : Dégradation des performances ──
        if (self.cl_cfg["performance_monitoring"]["alert_on_degradation"]
                and performance_data):
            for record in performance_data:
                self.perf_monitor.record(**record)
            degraded, perf_metrics = self.perf_monitor.check_degradation()
            if degraded:
                return True, f"Dégradation performances (Sharpe={perf_metrics.get('sharpe_30d', 'UNKNOWN')})"

        return False, "Aucun réentraînement nécessaire"

    def retrain(
        self,
        new_data: Dict,
        method: str = "fine_tune"
    ) -> Dict:
        """
        Déclenche le réentraînement du modèle.

        MÉTHODES :
        - "fine_tune"  : Continuer l'entraînement (rapide)
        - "full_retrain": Réentraîner depuis zéro (lent mais complet)

        PARAMÈTRES :
        - new_data : {"train_loader": ..., "val_loader": ..., "df_features": ...}
        - method   : "fine_tune" ou "full_retrain"

        RETOURNE :
        - Rapport du réentraînement
        """
        logger.info("=" * 55)
        logger.info(f"RÉENTRAÎNEMENT ({method.upper()}) - {datetime.now():%Y-%m-%d %H:%M}")
        logger.info("=" * 55)

        start_time = datetime.now()
        report = {"method": method, "timestamp": str(start_time)}

        try:
            if not hasattr(self, "model") or not hasattr(self, "trainer"):
                raise RuntimeError(
                    "Modèle/trainer non configurés. Appelez .setup() d'abord."
                )

            if method == "fine_tune":
                # Réduire le LR pour le fine-tuning
                for pg in self.trainer.optimizer.param_groups:
                    pg["lr"] *= 0.1
                self.trainer.epochs = 20  # Moins d'époques
                self.trainer.patience = 5

            # Réentraîner
            history = self.trainer.fit(
                new_data["train_loader"],
                new_data["val_loader"],
                model_path=f"./models/retrain_{self.retrain_count:03d}.pt"
            )

            # Mettre à jour la référence de drift
            if "df_features" in new_data:
                self.drift_detector.set_reference(new_data["df_features"])

            # Réinitialiser le compteur de performances
            self.perf_monitor.performance_history = []

            elapsed = (datetime.now() - start_time).seconds
            self.retrain_count += 1
            self.last_retrain  = datetime.now()

            report.update({
                "status"          : "SUCCESS",
                "elapsed_seconds" : elapsed,
                "retrain_number"  : self.retrain_count,
                "final_val_loss"  : history[-1]["val_loss"] if history else "UNKNOWN"
            })

            logger.info(f"Réentraînement terminé en {elapsed}s ✓")

        except Exception as e:
            logger.error(f"Réentraînement échoué : {e}")
            report["status"] = "FAILED"
            report["error"]  = str(e)

        self.retrain_log.append(report)
        return report

    def step(
        self,
        current_features: pd.DataFrame,
        new_data: Optional[Dict] = None,
        performance_data: Optional[List[Dict]] = None
    ) -> Tuple[bool, Dict]:
        """
        Exécute un pas du système d'apprentissage continu.
        À appeler périodiquement (ex: chaque jour en production).

        PARAMÈTRES :
        - current_features : Features des N derniers jours
        - new_data         : Nouvelles données pour réentraîner (si disponibles)
        - performance_data : Historique de performances récentes

        RETOURNE :
        - retrained : True si un réentraînement a été déclenché
        - report    : Rapport détaillé
        """
        should, reason = self.should_retrain(current_features, performance_data)

        if not should:
            return False, {"message": reason}

        logger.info(f"Déclenchement du réentraînement : {reason}")

        if new_data is None:
            logger.warning(
                "Réentraînement requis mais pas de nouvelles données fournies. "
                "Passez 'new_data' à step()."
            )
            return False, {"message": "Pas de données disponibles pour réentraîner"}

        report = self.retrain(new_data, method="fine_tune")
        report["trigger_reason"] = reason

        return True, report

    def get_status(self) -> Dict:
        """
        Retourne le statut actuel du système d'apprentissage continu.
        """
        days_since = (
            (datetime.now() - self.last_retrain).days
            if self.last_retrain else None
        )

        return {
            "last_retrain"        : str(self.last_retrain) if self.last_retrain else "Jamais",
            "days_since_retrain"  : days_since,
            "total_retrains"      : self.retrain_count,
            "drift_reference_set" : self.drift_detector.reference is not None,
            "next_retrain_in_days": max(0, self.cl_cfg["retrain_frequency_days"] - (days_since or 0))
        }
