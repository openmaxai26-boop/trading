"""
MICROSTRUCTURE DE MARCHÉ
=========================

QU'EST-CE QUE LA MICROSTRUCTURE ?
La microstructure de marché étudie les MÉCANISMES DÉTAILLÉS
par lesquels les prix se forment et les transactions s'effectuent.

C'est ce qui se passe "sous le capot" d'un marché :
- Comment les acheteurs et vendeurs interagissent
- Comment les prix bougent en réponse aux ordres
- Quels sont les niveaux de liquidité

FEATURES IMPLEMENTÉES :
1. Ratio Volume (participation vs habitude)
2. Pression d'achat/vente (order imbalance approximé)
3. Prix relatif dans la bougie (position dans le range)
4. Force relative jour/semaine/mois
5. Efficacité des prix (efficiency ratio)

NOTE POUR DÉBUTANTS :
La microstructure avancée nécessite les données du carnet d'ordres
(Level 2 data), qui sont coûteuses et complexes.
Ce module approxime ces métriques à partir des données OHLCV
disponibles gratuitement.
"""

import pandas as pd
import numpy as np
from ..utils.logger import get_logger

logger = get_logger("MicrostructureFeatures")


class MicrostructureFeatures:
    """
    Calcule les features de microstructure de marché.

    UTILISATION :
        msf = MicrostructureFeatures(config)
        df_enrichi = msf.compute_all(df_ohlcv)
    """

    def __init__(self, config: dict):
        self.config = config

    def price_position_in_range(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> pd.Series:
        """
        Calcule la position du prix de clôture dans le range de la bougie.

        DESCRIPTION :
        Mesure où le prix clôture par rapport au Haut et Bas de la journée.

        FORMULE :
        Position = (Clôture - Bas) / (Haut - Bas)

        INTERPRÉTATION :
        - Position = 1.0 : Clôture au plus haut → Signal haussier fort
        - Position = 0.5 : Clôture au milieu → Indécision
        - Position = 0.0 : Clôture au plus bas → Signal baissier fort

        CET INDICATEUR CAPTURE :
        La "persistance" des acheteurs/vendeurs jusqu'à la fin de séance.
        """
        daily_range = high - low
        position = (close - low) / (daily_range + 1e-10)
        return position.clip(0, 1).rename("price_position")

    def order_flow_imbalance(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """
        Approximation du déséquilibre du flux d'ordres.

        DESCRIPTION :
        L'ordre imbalance mesure si les acheteurs ou vendeurs
        dominent un marché. En finance quantitative, on calcule
        normalement ceci avec les données tick-by-tick.

        APPROXIMATION AVEC OHLCV (méthode de Tick Rule) :
        Si Clôture > Ouverture → Pression acheteuse
        Si Clôture < Ouverture → Pression vendeuse

        On pondère par le volume et la taille du mouvement.

        FORMULE :
        Imbalance = sign(Close - Open) × Volume × |Close - Open| / Range

        VALEURS :
        - Positive : Pression acheteuse dominante
        - Négative : Pression vendeuse dominante
        - ~0 : Équilibre entre acheteurs et vendeurs
        """
        direction = np.sign(close - open_)
        move_size = (close - open_).abs()
        daily_range = (high - low).clip(lower=1e-10)

        # Volume pondéré par direction et taille relative du mouvement
        imbalance = direction * volume * (move_size / daily_range)

        # Normaliser par le volume moyen pour comparabilité
        vol_ma = volume.rolling(20).mean().replace(0, np.nan)
        normalized_imbalance = imbalance / vol_ma.fillna(1)

        return normalized_imbalance.rename("order_imbalance")

    def liquidity_pressure(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        window: int = 20
    ) -> pd.DataFrame:
        """
        Indicateur de pression de liquidité.

        DESCRIPTION :
        Mesure si le volume est anormalement élevé ou faible
        par rapport à l'habituel, et dans quelle direction.

        COMPOSANTS :
        1. Volume Ratio : Volume actuel / Volume moyen
        2. Price Impact : Impact du volume sur le prix ($ par unité)

        POURQUOI C'EST IMPORTANT ?
        - Volume anormalement élevé avec hausse de prix
          → Forte conviction des acheteurs → Continuation probable
        - Volume anormalement élevé avec baisse de prix
          → Capitulation (vente panique) → Possible retournement

        PARAMÈTRES :
        - window : Fenêtre pour calculer la moyenne du volume
        """
        # Ratio volume actuel / moyenne
        vol_ma = volume.rolling(window).mean().replace(0, np.nan)
        vol_ratio = volume / vol_ma.fillna(1)

        # Impact du prix : amplitude de la bougie / volume (spread coût)
        daily_range = (high - low).clip(lower=1e-10)
        price_impact = daily_range / volume.clip(lower=1)

        # Normaliser le price impact
        price_impact_norm = price_impact / price_impact.rolling(window).mean().replace(0, np.nan)

        return pd.DataFrame({
            "volume_ratio": vol_ratio,
            "price_impact": price_impact_norm.fillna(1)
        })

    def efficiency_ratio(
        self,
        close: pd.Series,
        window: int = 10
    ) -> pd.Series:
        """
        Ratio d'efficacité de Perry Kaufman.

        DESCRIPTION :
        Mesure l'efficacité du mouvement de prix.
        Compare le mouvement NET (début → fin) au mouvement TOTAL
        (somme de tous les petits déplacements).

        FORMULE :
        ER = |Prix[t] - Prix[t-n]| / Σ|Prix[i] - Prix[i-1]|

        INTERPRÉTATION :
        - ER proche de 1 : Tendance directionnelle très forte
                           (le prix va dans une seule direction)
        - ER proche de 0 : Marché chaotique, sans direction
                           (le prix oscille sans avancer)

        UTILISATION :
        Utile pour adapter la stratégie :
        - ER élevé → stratégie tendancielle
        - ER bas   → stratégie de mean-reversion

        PARAMÈTRES :
        - window : Période d'observation (défaut 10 jours)
        """
        # Mouvement net sur la période
        net_movement = (close - close.shift(window)).abs()

        # Somme des mouvements journaliers absolus
        daily_changes = close.diff().abs()
        total_movement = daily_changes.rolling(window).sum().clip(lower=1e-10)

        er = net_movement / total_movement
        return er.clip(0, 1).rename(f"efficiency_ratio_{window}d")

    def volume_profile(
        self,
        close: pd.Series,
        volume: pd.Series,
        window: int = 20
    ) -> pd.DataFrame:
        """
        Profile de volume simplifié.

        DESCRIPTION :
        Mesure à quels niveaux de prix le volume se concentre.
        C'est une approximation du Volume Profile institutionnel.

        CONCEPT :
        Les niveaux avec beaucoup de volume = niveaux de support/résistance
        (beaucoup d'acteurs ont des positions à ces prix)

        MÉTRIQUES :
        1. Proximity to POC (Point of Control) : Distance au prix le + échangé
        2. Volume at current price level : Volume relatif au niveau actuel

        PARAMÈTRES :
        - window : Fenêtre en jours (défaut 20)
        """
        # VWAP comme proxy du Point of Control
        typical_price = close  # Simplification avec close
        weighted_price = typical_price * volume

        vwap_rolling = (
            weighted_price.rolling(window).sum() /
            volume.rolling(window).sum().clip(lower=1e-10)
        )

        # Distance au VWAP roulant (proxy du POC)
        distance_to_poc = (close - vwap_rolling) / (vwap_rolling + 1e-10)

        # Z-score du volume (volume anormal ?)
        vol_mean = volume.rolling(window).mean()
        vol_std = volume.rolling(window).std().clip(lower=1e-10)
        vol_zscore = (volume - vol_mean) / vol_std

        return pd.DataFrame({
            "distance_to_vwap": distance_to_poc,
            "volume_zscore": vol_zscore.clip(-3, 3)
        })

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule toutes les features de microstructure.

        PARAMÈTRES :
        - df : DataFrame avec colonnes OHLCV (open, high, low, close, volume)

        RETOURNE :
        - DataFrame enrichi avec features de microstructure
        """
        result = df.copy()

        try:
            # Position dans le range
            result["price_position"] = self.price_position_in_range(
                df["high"], df["low"], df["close"]
            )

            # Déséquilibre du flux d'ordres
            result["order_imbalance"] = self.order_flow_imbalance(
                df["open"], df["high"], df["low"], df["close"], df["volume"]
            )

            # Pression de liquidité
            liq_df = self.liquidity_pressure(
                df["high"], df["low"], df["close"], df["volume"]
            )
            result = pd.concat([result, liq_df], axis=1)

            # Ratio d'efficacité sur plusieurs périodes
            for w in [5, 10, 20]:
                result[f"efficiency_ratio_{w}d"] = self.efficiency_ratio(
                    df["close"], window=w
                )

            # Volume profile
            vp_df = self.volume_profile(df["close"], df["volume"])
            result = pd.concat([result, vp_df], axis=1)

            # Gap overnight (écart entre clôture j-1 et ouverture j)
            result["overnight_gap"] = (
                (df["open"] - df["close"].shift(1)) /
                df["close"].shift(1).clip(lower=1e-10)
            )

            # Force du trend intraday
            result["intraday_trend"] = (
                (df["close"] - df["open"]) /
                (df["high"] - df["low"]).clip(lower=1e-10)
            )

            logger.debug(f"Features microstructure calculées : {len(result.columns)} colonnes")

        except Exception as e:
            logger.error(f"Erreur calcul features microstructure : {e}")
            import traceback
            logger.debug(traceback.format_exc())

        return result
