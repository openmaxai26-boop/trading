"""
FEATURES STATISTIQUES
======================

QU'EST-CE QUE LES FEATURES STATISTIQUES ?
Au-delà des indicateurs techniques classiques, les statisticiens
quantitatifs utilisent des métriques plus avancées pour capturer
la DISTRIBUTION des rendements.

CES FEATURES RÉPONDENT AUX QUESTIONS :
- Quelle est la volatilité récente ? (risque)
- Les mouvements sont-ils asymétriques ? (skewness)
- Y a-t-il des mouvements extrêmes inhabituels ? (kurtosis)
- Quel est le risque ajusté ? (Sharpe ratio roulant)
- Y a-t-il une persistance de la volatilité ? (GARCH-like)

GLOSSAIRE POUR DÉBUTANTS :
- Rendement : Variation du prix en % (ex: +2%)
- Volatilité : Écart-type des rendements (mesure du risque)
- Skewness   : Asymétrie de la distribution des rendements
               Positive = plus de gros gains | Négative = plus de grosses pertes
- Kurtosis   : "Queues épaisses" = probabilité d'événements extrêmes
               > 3 = distribution "leptokurtique" (queues lourdes)
- Sharpe     : Rendement / Risque (plus c'est élevé, mieux c'est)
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Optional
from ..utils.logger import get_logger

logger = get_logger("StatisticalFeatures")


class StatisticalFeatures:
    """
    Calcule les features statistiques des séries temporelles financières.

    UTILISATION :
        sf = StatisticalFeatures(config)
        df_enrichi = sf.compute_all(df_ohlcv)
    """

    def __init__(self, config: dict):
        self.cfg = config["features"]["statistical"]

    def log_returns(
        self,
        close: pd.Series,
        periods: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Calcule les rendements logarithmiques sur différentes périodes.

        POURQUOI LES RENDEMENTS LOGARITHMIQUES ?
        Avantages des log-rendements vs rendements simples :
        1. Symétriques : -50% et +100% ont le même impact
        2. Additifs dans le temps : ret_1 + ret_2 = ret_total
        3. Meilleure propriété statistique (plus proches d'une normale)

        FORMULE :
        r_t = ln(P_t / P_(t-n)) = ln(P_t) - ln(P_(t-n))

        EXEMPLE :
        Prix hier = 100$, aujourd'hui = 102$
        Rendement simple = (102-100)/100 = 2%
        Log-rendement = ln(102/100) = 1.98%
        → Très proche pour les petits rendements

        PARAMÈTRES :
        - close   : Série des prix de clôture
        - periods : Liste de périodes (ex: [1, 5, 10, 21] = 1j, 1sem, 2sem, 1mois)
        """
        periods = periods or self.cfg["returns_periods"]
        result = {}

        for p in periods:
            ret = np.log(close / close.shift(p))
            result[f"return_{p}d"] = ret

        return pd.DataFrame(result)

    def rolling_volatility(
        self,
        close: pd.Series,
        windows: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Calcule la volatilité réalisée sur différentes fenêtres.

        QU'EST-CE QUE LA VOLATILITÉ ?
        La volatilité = écart-type des rendements annualisé.
        C'est la mesure principale du RISQUE en finance.

        FORMULE :
        Vol_t = Std(r_t, ..., r_(t-n)) × √252
        (×√252 pour annualiser : 252 jours de bourse par an)

        EXEMPLE :
        Écart-type journalier = 1%
        Volatilité annualisée = 1% × √252 = 15.87%

        INTERPRÉTATION :
        - Vol < 15% : Actif peu volatile (obligation, or)
        - Vol 15-30% : Actif normalement volatile (action)
        - Vol > 30% : Actif très volatile (crypto, small cap)
        - Vol > 80% : Extrêmement volatile (altcoins)

        PARAMÈTRES :
        - windows : Fenêtres en jours (ex: [5, 21, 63])
        """
        windows = windows or [5, 21, 63]
        returns = np.log(close / close.shift(1))
        result = {}

        for w in windows:
            # ×√252 pour annualiser (252 jours ouvrables par an)
            vol = returns.rolling(window=w).std() * np.sqrt(252)
            result[f"vol_{w}d"] = vol

        # Ratio de volatilité court terme / long terme
        if 5 in windows and 21 in windows:
            result["vol_ratio_5_21"] = result["vol_5d"] / (result["vol_21d"] + 1e-10)

        return pd.DataFrame(result)

    def rolling_skewness(
        self,
        close: pd.Series,
        window: int = 21
    ) -> pd.Series:
        """
        Calcule la skewness (asymétrie) des rendements sur fenêtre glissante.

        QU'EST-CE QUE LA SKEWNESS ?
        La skewness mesure l'asymétrie de la distribution des rendements.

        VISUALISATION :
        Distribution symétrique (skew=0) :
            ████
          ██████
        ████████████
        ─────────────
         Pertes  Gains

        Skewness positive (skew>0) : Queue à droite (rares gros gains)
        Skewness négative (skew<0) : Queue à gauche (rares grosses pertes)

        IMPORTANCE POUR LE TRADING :
        - Les actions ont souvent une skewness négative
          (les krachs sont plus violents que les hausses)
        - Reconnaître une skewness positive peut identifier des
          actifs avec un potentiel de hausse asymétrique

        PARAMÈTRES :
        - window : Fenêtre glissante en jours (défaut 21 = 1 mois)
        """
        returns = np.log(close / close.shift(1))

        def safe_skew(x):
            """Calcule le skew en gérant les erreurs."""
            if len(x.dropna()) < 3:
                return np.nan
            return float(stats.skew(x.dropna()))

        skewness = returns.rolling(window=window).apply(safe_skew, raw=False)
        return skewness.rename(f"skewness_{window}d")

    def rolling_kurtosis(
        self,
        close: pd.Series,
        window: int = 21
    ) -> pd.Series:
        """
        Calcule la kurtosis (aplatissement) des rendements.

        QU'EST-CE QUE LA KURTOSIS ?
        La kurtosis mesure la "lourdeur des queues" de distribution.

        INTERPRÉTATION :
        - Kurtosis = 3 : Distribution normale (référence)
        - Kurtosis > 3 : Queues lourdes = plus d'événements extrêmes
                         qu'une distribution normale (typique des marchés !)
        - Kurtosis < 3 : Queues légères = peu d'extrêmes

        POURQUOI C'EST CRUCIAL EN FINANCE ?
        Les marchés financiers ont une kurtosis >> 3 (souvent 5-10).
        Cela signifie que les krachs et rallyes violents sont
        BIEN PLUS FRÉQUENTS que ce que la distribution normale prédit.
        C'est ce qu'on appelle les "fat tails" (queues épaisses).

        PARAMÈTRES :
        - window : Fenêtre glissante en jours
        """
        returns = np.log(close / close.shift(1))

        def safe_kurt(x):
            """Calcule la kurtosis en gérant les erreurs."""
            if len(x.dropna()) < 4:
                return np.nan
            return float(stats.kurtosis(x.dropna()))

        kurt = returns.rolling(window=window).apply(safe_kurt, raw=False)
        return kurt.rename(f"kurtosis_{window}d")

    def rolling_sharpe(
        self,
        close: pd.Series,
        window: int = 21,
        risk_free_rate: float = 0.05
    ) -> pd.Series:
        """
        Calcule le ratio de Sharpe roulant.

        QU'EST-CE QUE LE RATIO DE SHARPE ?
        Le Sharpe mesure le rendement AJUSTÉ au risque.
        C'est l'indicateur de performance le plus utilisé en finance.

        FORMULE :
        Sharpe = (Rendement - Taux Sans Risque) / Volatilité

        INTERPRÉTATION :
        - Sharpe < 0   : Performance inférieure au taux sans risque
        - Sharpe 0-0.5 : Médiocre
        - Sharpe 0.5-1 : Acceptable
        - Sharpe 1-2   : Bon
        - Sharpe > 2   : Excellent (rare)
        - Sharpe > 3   : Exceptionnel (hedge funds top)

        EXEMPLE :
        Rendement annuel = 15%, Volatilité = 10%, Taux sans risque = 5%
        Sharpe = (15% - 5%) / 10% = 1.0 → Bon!

        PARAMÈTRES :
        - window          : Fenêtre glissante en jours
        - risk_free_rate  : Taux sans risque annualisé (défaut 5%)
        """
        returns = np.log(close / close.shift(1))
        daily_rf = risk_free_rate / 252  # Taux journalier

        rolling_mean = returns.rolling(window=window).mean()
        rolling_std = returns.rolling(window=window).std()

        sharpe = (rolling_mean - daily_rf) / (rolling_std + 1e-10) * np.sqrt(252)
        return sharpe.rename(f"sharpe_{window}d")

    def rolling_max_drawdown(
        self,
        close: pd.Series,
        window: int = 63
    ) -> pd.Series:
        """
        Calcule le drawdown maximum sur une fenêtre glissante.

        QU'EST-CE QU'UN DRAWDOWN ?
        Le drawdown mesure la chute depuis le dernier sommet.

        EXEMPLE :
        Prix : 100 → 120 → 90 → 110
        Sommet : 120
        Drawdown depuis sommet : (90 - 120) / 120 = -25%

        POURQUOI C'EST IMPORTANT ?
        Le max drawdown est la mesure de risque préférée des
        investisseurs. Elle répond à la question :
        "Au pire, combien aurais-je perdu ?"

        OBJECTIF DU SYSTÈME :
        Limiter le drawdown à -15% maximum (configuré dans config.yaml)

        PARAMÈTRES :
        - window : Fenêtre en jours (défaut 63 = 3 mois)
        """
        def max_dd(prices):
            """Calcule le max drawdown sur une série de prix."""
            if len(prices) == 0:
                return 0.0
            cummax = np.maximum.accumulate(prices)
            drawdowns = (prices - cummax) / (cummax + 1e-10)
            return float(np.min(drawdowns))

        mdd = close.rolling(window=window).apply(max_dd, raw=True)
        return mdd.rename(f"max_drawdown_{window}d")

    def volatility_clustering(self, close: pd.Series, window: int = 21) -> pd.Series:
        """
        Détecte le regroupement de la volatilité (clustering).

        QU'EST-CE QUE LE CLUSTERING DE VOLATILITÉ ?
        En finance, les périodes de forte volatilité ont tendance
        à se regrouper : "La volatilité est volatile elle-même."

        C'est le principe de base des modèles GARCH.
        "Les grandes variations sont suivies de grandes variations."

        COMMENT LE MESURER ?
        On calcule l'autocorrélation des rendements au CARRÉ.
        Si l'autocorrélation est forte → clustering présent.

        PARAMÈTRE :
        - window : Fenêtre d'analyse (défaut 21 jours)
        """
        returns = np.log(close / close.shift(1)).fillna(0)
        sq_returns = returns ** 2  # Rendements au carré

        # Autocorrélation lag-1 des rendements au carré
        autocorr = sq_returns.rolling(window=window).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else 0,
            raw=False
        )
        return autocorr.rename("vol_clustering")

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule toutes les features statistiques.

        PARAMÈTRES :
        - df : DataFrame avec colonnes OHLCV

        RETOURNE :
        - DataFrame enrichi avec features statistiques
        """
        result = df.copy()

        try:
            # Rendements sur plusieurs périodes
            returns_df = self.log_returns(df["close"])
            result = pd.concat([result, returns_df], axis=1)

            # Volatilité sur plusieurs fenêtres
            vol_df = self.rolling_volatility(df["close"])
            result = pd.concat([result, vol_df], axis=1)

            # Métriques distributionnelles
            result["skewness_21d"] = self.rolling_skewness(df["close"], 21)
            result["kurtosis_21d"] = self.rolling_kurtosis(df["close"], 21)

            # Performance ajustée au risque
            result["sharpe_21d"] = self.rolling_sharpe(df["close"], 21)
            result["sharpe_63d"] = self.rolling_sharpe(df["close"], 63)

            # Drawdown
            result["max_drawdown_63d"] = self.rolling_max_drawdown(df["close"], 63)

            # Clustering de volatilité
            result["vol_clustering"] = self.volatility_clustering(df["close"])

            # Momentum (tendance)
            for p in [5, 21, 63]:
                result[f"momentum_{p}d"] = (
                    df["close"] / df["close"].shift(p) - 1
                ).rename(f"momentum_{p}d")

            # Mean reversion score (distance à la moyenne)
            for p in [20, 50]:
                sma = df["close"].rolling(p).mean()
                result[f"mean_reversion_{p}d"] = (df["close"] - sma) / (sma + 1e-10)

            logger.debug(f"Features statistiques calculées : {len(result.columns)} colonnes")

        except Exception as e:
            logger.error(f"Erreur calcul features statistiques : {e}")
            import traceback
            logger.debug(traceback.format_exc())

        return result
