"""
INDICATEURS TECHNIQUES
========================

QU'EST-CE QU'UN INDICATEUR TECHNIQUE ?
Un indicateur technique est un calcul mathématique basé sur
l'historique des prix et volumes qui aide à :
- Identifier la TENDANCE (hausse, baisse, latéral)
- Mesurer la FORCE du mouvement
- Détecter les zones de SURACHAT / SURVENTE

INDICATEURS IMPLÉMENTÉS :
1. RSI   - Force relative des mouvements
2. MACD  - Convergence/Divergence des moyennes mobiles
3. Bandes de Bollinger - Volatilité et prix extrêmes
4. VWAP  - Prix moyen pondéré par volume
5. ATR   - Vraie amplitude du mouvement (volatilité)
6. EMA   - Moyenne mobile exponentielle

IMPORTANT : Ces indicateurs sont des FEATURES pour l'IA.
Le modèle va apprendre quels indicateurs sont importants
et comment les combiner pour prédire.
"""

import pandas as pd
import numpy as np
from typing import Optional
from ..utils.logger import get_logger

logger = get_logger("TechnicalFeatures")


class TechnicalFeatures:
    """
    Calcule tous les indicateurs techniques à partir des données OHLCV.

    UTILISATION :
        tf = TechnicalFeatures(config)
        df_with_indicators = tf.compute_all(df_ohlcv)
    """

    def __init__(self, config: dict):
        self.cfg = config["features"]["technical"]

    def rsi(self, close: pd.Series, period: Optional[int] = None) -> pd.Series:
        """
        RSI (Relative Strength Index) - Indice de Force Relative

        DESCRIPTION :
        Le RSI mesure la vitesse et le changement des mouvements de prix.
        Créé par J. Welles Wilder en 1978, c'est l'un des indicateurs
        les plus utilisés au monde.

        INTERPRÉTATION :
        - RSI > 70 : Zone de SURACHAT → risque de retournement baissier
        - RSI < 30 : Zone de SURVENTE → possible rebond haussier
        - RSI entre 30-70 : Zone neutre

        FORMULE :
        RSI = 100 - (100 / (1 + RS))
        RS = Moyenne des hausses / Moyenne des baisses sur N jours

        EXEMPLE :
        Sur 14 jours, si en moyenne le prix monte de 1.5% les jours
        haussiers et baisse de 0.8% les jours baissiers :
        RS = 1.5 / 0.8 = 1.875
        RSI = 100 - (100 / (1 + 1.875)) = 65.2

        PARAMÈTRES :
        - close  : Série des prix de clôture
        - period : Période (défaut 14 jours)
        """
        period = period or self.cfg["rsi_period"]

        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        # Moyenne mobile exponentielle des gains et pertes
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

        rs = avg_gain / (avg_loss + 1e-10)  # +1e-10 pour éviter division par zéro
        rsi = 100 - (100 / (1 + rs))

        return rsi.rename("rsi")

    def macd(
        self,
        close: pd.Series,
        fast: Optional[int] = None,
        slow: Optional[int] = None,
        signal: Optional[int] = None
    ) -> pd.DataFrame:
        """
        MACD (Moving Average Convergence Divergence)
        Convergence/Divergence des Moyennes Mobiles

        DESCRIPTION :
        Le MACD montre la relation entre deux moyennes mobiles
        exponentielles (EMA) du prix.
        Développé par Gerald Appel dans les années 1970.

        COMPOSANTS :
        1. MACD Line     = EMA(12) - EMA(26)
           → Différence entre EMA rapide et EMA lente
        2. Signal Line   = EMA(9) de la MACD Line
           → Lissage du MACD
        3. Histogramme   = MACD Line - Signal Line
           → Force du signal

        SIGNAUX DE TRADING :
        - MACD croise au-dessus du Signal → Signal HAUSSIER
        - MACD croise en-dessous du Signal → Signal BAISSIER
        - Histogramme augmente → Momentum haussier s'accélère

        PARAMÈTRES :
        - fast   : Période EMA rapide (défaut 12)
        - slow   : Période EMA lente (défaut 26)
        - signal : Période EMA du signal (défaut 9)
        """
        fast = fast or self.cfg["macd_fast"]
        slow = slow or self.cfg["macd_slow"]
        signal = signal or self.cfg["macd_signal"]

        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return pd.DataFrame({
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_hist": histogram
        })

    def bollinger_bands(
        self,
        close: pd.Series,
        period: Optional[int] = None,
        std_dev: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Bandes de Bollinger

        DESCRIPTION :
        Les bandes de Bollinger entourent le prix avec des bandes
        basées sur la volatilité. Créées par John Bollinger.

        COMPOSANTS :
        1. Bande Supérieure = SMA(20) + 2 × Écart-type(20)
        2. Bande Médiane    = SMA(20) (Moyenne mobile simple)
        3. Bande Inférieure = SMA(20) - 2 × Écart-type(20)
        4. Largeur          = (Supérieure - Inférieure) / Médiane
        5. %B               = (Prix - Inférieure) / (Supérieure - Inférieure)

        INTERPRÉTATION :
        - Prix touche la bande sup : zone de résistance possible
        - Prix touche la bande inf : zone de support possible
        - Bandes se resserrent (squeeze) → explosion de volatilité à venir
        - %B > 1 : Prix au-dessus de la bande sup (surachat)
        - %B < 0 : Prix en-dessous de la bande inf (survente)

        PARAMÈTRES :
        - period  : Période de la SMA (défaut 20)
        - std_dev : Multiplicateur d'écart-type (défaut 2.0)
        """
        period = period or self.cfg["bb_period"]
        std_dev = std_dev or self.cfg["bb_std"]

        sma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()

        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        width = (upper - lower) / (sma + 1e-10)
        pct_b = (close - lower) / (upper - lower + 1e-10)

        return pd.DataFrame({
            "bb_upper": upper,
            "bb_middle": sma,
            "bb_lower": lower,
            "bb_width": width,
            "bb_pct": pct_b
        })

    def vwap(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: Optional[int] = None
    ) -> pd.Series:
        """
        VWAP (Volume Weighted Average Price)
        Prix Moyen Pondéré par le Volume

        DESCRIPTION :
        Le VWAP calcule le prix moyen auquel les transactions
        ont eu lieu, pondéré par le volume.

        FORMULE :
        VWAP = Σ(Prix Typique × Volume) / Σ(Volume)
        Prix Typique = (Haut + Bas + Clôture) / 3

        UTILISATION :
        - Les institutionnels (grands fonds) utilisent le VWAP
          comme référence d'exécution
        - Prix > VWAP → marché haussier intraday
        - Prix < VWAP → marché baissier intraday
        - Idéal pour données intraday (1h, 15min)

        RETOURNE :
        - VWAP sur la période glissante
        - Distance du prix au VWAP (en %)
        """
        period = period or self.cfg.get("vwap_period", 14)

        typical_price = (high + low + close) / 3
        tp_x_vol = typical_price * volume

        vwap_values = (
            tp_x_vol.rolling(window=period).sum() /
            volume.rolling(window=period).sum()
        )

        return vwap_values.rename("vwap")

    def atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: Optional[int] = None
    ) -> pd.Series:
        """
        ATR (Average True Range) - Vraie Amplitude Moyenne

        DESCRIPTION :
        L'ATR mesure la volatilité d'un actif en calculant
        la "vraie amplitude" de chaque bougie.

        VRAIE AMPLITUDE (True Range) = max de :
        1. Haut - Bas (amplitude normale)
        2. |Haut - Clôture précédente| (gap haussier)
        3. |Bas - Clôture précédente| (gap baissier)

        UTILISATION :
        - Calibrer les stop-loss (ex: stop = prix - 2×ATR)
        - Mesurer la volatilité courante
        - Comparer la volatilité entre actifs différents

        EXEMPLE :
        ATR de 3$ sur AAPL → le prix varie en moyenne de 3$
        par jour → stop-loss à 6$ sous le point d'entrée (2×ATR)
        """
        period = period or self.cfg.get("atr_period", 14)

        high_low = high - low
        high_close = (high - close.shift(1)).abs()
        low_close = (low - close.shift(1)).abs()

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr_values = true_range.ewm(span=period, adjust=False).mean()

        return atr_values.rename("atr")

    def ema(self, close: pd.Series, period: int) -> pd.Series:
        """
        EMA (Exponential Moving Average) - Moyenne Mobile Exponentielle

        DESCRIPTION :
        L'EMA donne plus de poids aux données récentes,
        contrairement à la SMA (simple) qui traite tous les
        points équitablement.

        FORMULE :
        EMA_t = α × Prix_t + (1-α) × EMA_(t-1)
        α = 2 / (période + 1)

        EXEMPLE (période 10) :
        α = 2/(10+1) = 0.18
        Aujourd'hui compte pour 18%, hier pour 82% × 18%, etc.

        PARAMÈTRES :
        - close  : Série des prix de clôture
        - period : Période (ex: 20, 50, 200)
        """
        return close.ewm(span=period, adjust=False).mean().rename(f"ema_{period}")

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule TOUS les indicateurs techniques et les ajoute au DataFrame.

        PARAMÈTRES :
        - df : DataFrame avec colonnes open, high, low, close, volume

        RETOURNE :
        - DataFrame enrichi avec tous les indicateurs techniques
        """
        result = df.copy()

        try:
            # RSI
            result["rsi"] = self.rsi(df["close"])

            # MACD
            macd_df = self.macd(df["close"])
            result = pd.concat([result, macd_df], axis=1)

            # Bandes de Bollinger
            bb_df = self.bollinger_bands(df["close"])
            result = pd.concat([result, bb_df], axis=1)

            # VWAP
            result["vwap"] = self.vwap(df["high"], df["low"], df["close"], df["volume"])

            # Distance VWAP/Prix en %
            result["vwap_distance"] = (df["close"] - result["vwap"]) / (result["vwap"] + 1e-10)

            # ATR
            result["atr"] = self.atr(df["high"], df["low"], df["close"])

            # ATR normalisé (ATR / Prix) pour comparaison entre actifs
            result["atr_pct"] = result["atr"] / (df["close"] + 1e-10)

            # EMA multiples (tendances court/moyen/long terme)
            for period in [9, 20, 50, 200]:
                result[f"ema_{period}"] = self.ema(df["close"], period)

            # Croisements EMA (signaux de tendance)
            result["ema_cross_9_20"] = np.sign(result["ema_9"] - result["ema_20"])
            result["ema_cross_20_50"] = np.sign(result["ema_20"] - result["ema_50"])
            result["ema_cross_50_200"] = np.sign(result["ema_50"] - result["ema_200"])

            # Ratio volume vs moyenne mobile du volume
            result["volume_ratio"] = (
                df["volume"] / df["volume"].rolling(20).mean().replace(0, np.nan)
            )

            logger.debug(f"Indicateurs techniques calculés : {len(result.columns)} colonnes")

        except Exception as e:
            logger.error(f"Erreur calcul indicateurs techniques : {e}")
            import traceback
            logger.debug(traceback.format_exc())

        return result
