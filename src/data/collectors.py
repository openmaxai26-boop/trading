"""
COUCHE 1 : COLLECTE DES DONNÉES DE MARCHÉ
==========================================

QU'EST-CE QUE CE MODULE FAIT ?
Ce module télécharge les données financières depuis plusieurs sources :
- Yahoo Finance : actions, ETF, commodités (GRATUIT)
- CCXT : données crypto depuis les exchanges (Binance, etc.)

QUELLES DONNÉES COLLECTE-T-ON ?
O = Open  : Prix d'ouverture de la bougie
H = High  : Prix le plus haut atteint
L = Low   : Prix le plus bas atteint
C = Close : Prix de clôture
V = Volume: Volume d'échanges

POURQUOI 2 ANS D'HISTORIQUE ?
- Couvrir un cycle de marché complet (bull + bear + range)
- Avoir assez de données pour entraîner les modèles IA
- Valider les résultats sur des données non vues

ARCHITECTURE :
DataCollector
├── download_stock_data()    → Yahoo Finance (actions/ETF)
├── download_crypto_data()   → CCXT (Bitcoin, Ethereum...)
├── download_macro_data()    → Indicateurs macro (taux, inflation)
└── get_all_data()           → Collecte tout en une fois
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import time

from ..utils.logger import get_logger

logger = get_logger("DataCollector")


class MarketDataCollector:
    """
    Collecteur principal de données de marché.

    COMMENT L'UTILISER :
        collector = MarketDataCollector(config)
        data = collector.get_all_data()
        # data["AAPL"] → DataFrame avec colonnes Open, High, Low, Close, Volume

    PARAMÈTRES DU CONSTRUCTEUR :
    - config : Dictionnaire de configuration (chargé depuis config.yaml)
    """

    def __init__(self, config: dict):
        self.config = config
        self.cache_dir = Path(config["data"]["cache_dir"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.lookback_days = config["timeframes"]["lookback_days"]
        logger.info(f"DataCollector initialisé | Cache : {self.cache_dir}")

    def download_stock_data(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Télécharge les données OHLCV pour les actions et ETF via Yahoo Finance.

        COMMENT ÇA MARCHE :
        Yahoo Finance est une source GRATUITE de données financières.
        yfinance est une bibliothèque Python qui interroge Yahoo Finance automatiquement.

        PARAMÈTRES :
        - symbols   : Liste de symboles boursiers (["AAPL", "MSFT", "SPY"])
        - start_date: Date de début (format "YYYY-MM-DD")
        - end_date  : Date de fin (format "YYYY-MM-DD")
        - interval  : Granularité des données ("1d"=journalier, "1h"=horaire)

        RETOURNE :
        - Dictionnaire {symbole: DataFrame avec colonnes OHLCV}
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance non installé. Lancez : pip install yfinance")
            raise

        if start_date is None:
            start_date = (
                datetime.now() - timedelta(days=self.lookback_days)
            ).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        data = {}

        for symbol in symbols:
            logger.info(f"Téléchargement de {symbol} depuis Yahoo Finance...")

            # Vérifier le cache local d'abord
            cache_file = self.cache_dir / f"{symbol}_{interval}_{start_date}_{end_date}.parquet"
            if cache_file.exists():
                logger.info(f"  → {symbol} : chargé depuis le cache local")
                data[symbol] = pd.read_parquet(cache_file)
                continue

            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    auto_adjust=True    # Ajustement automatique des splits/dividendes
                )

                if df.empty:
                    logger.warning(f"  → {symbol} : aucune donnée reçue (symbole invalide ?)")
                    continue

                # Standardiser les noms de colonnes
                df.columns = [c.lower() for c in df.columns]
                df = df[["open", "high", "low", "close", "volume"]].copy()
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()

                # Sauvegarder en cache
                df.to_parquet(cache_file)

                data[symbol] = df
                logger.info(f"  → {symbol} : {len(df)} bougies téléchargées ✓")

                # Pause pour éviter de surcharger l'API
                time.sleep(0.3)

            except Exception as e:
                logger.error(f"  → {symbol} : Erreur de téléchargement : {e}")

        return data

    def download_crypto_data(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Télécharge les données crypto.

        STRATÉGIE :
        On utilise Yahoo Finance pour les cryptos (BTC-USD, ETH-USD).
        C'est gratuit et suffisant pour commencer.
        Pour des données en temps réel plus précises, on utiliserait CCXT + Binance.

        POURQUOI YAHOO FINANCE POUR LA CRYPTO ?
        Binance et CCXT nécessitent une clé API.
        Yahoo Finance donne des données OHLCV journalières sans clé.
        """
        logger.info(f"Téléchargement des données crypto : {symbols}")
        # Yahoo Finance gère nativement les paires crypto (BTC-USD, ETH-USD)
        return self.download_stock_data(symbols, start_date, end_date, interval)

    def download_macro_data(self) -> pd.DataFrame:
        """
        Collecte les données macro-économiques via Yahoo Finance.

        DONNÉES MACRO DISPONIBLES GRATUITEMENT :
        - ^TNX : Rendement des bons du Trésor américain 10 ans
        - ^VIX : Indice de volatilité (peur du marché)
        - DX-Y.NYB : Indice Dollar américain

        CES DONNÉES SONT IMPORTANTES PARCE QUE :
        - Hausse des taux → baisse des actions en général
        - VIX élevé → peur du marché → opportunités ou dangers
        - Dollar fort → pression sur les matières premières
        """
        macro_symbols = {
            "^TNX": "taux_10ans",
            "^VIX": "vix",
            "DX-Y.NYB": "dollar_index",
        }

        logger.info("Téléchargement des données macro-économiques...")
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance non installé")
            return pd.DataFrame()

        frames = {}
        for symbol, name in macro_symbols.items():
            try:
                df = yf.download(
                    symbol,
                    period=f"{self.lookback_days}d",
                    interval="1d",
                    auto_adjust=True,
                    progress=False
                )
                if not df.empty:
                    frames[name] = df["Close"]
                    logger.info(f"  → {name} ({symbol}) : {len(df)} points ✓")
            except Exception as e:
                logger.warning(f"  → {name} : impossible de télécharger : {e}")

        if frames:
            macro_df = pd.DataFrame(frames)
            macro_df.index = pd.to_datetime(macro_df.index)
            return macro_df

        return pd.DataFrame()

    def get_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Point d'entrée principal : collecte TOUTES les données configurées.

        RETOURNE :
        Un dictionnaire avec tous les actifs :
        {
            "AAPL": DataFrame(Open, High, Low, Close, Volume),
            "BTC-USD": DataFrame(Open, High, Low, Close, Volume),
            "GC=F": DataFrame(Open, High, Low, Close, Volume),
            ...
        }
        """
        logger.info("=" * 60)
        logger.info("DÉMARRAGE DE LA COLLECTE COMPLÈTE DES DONNÉES")
        logger.info("=" * 60)

        all_data = {}

        # 1. Actions
        stock_symbols = self.config["assets"]["stocks"]
        stock_data = self.download_stock_data(stock_symbols)
        all_data.update(stock_data)

        # 2. Commodités
        commodity_symbols = self.config["assets"]["commodities"]
        commodity_data = self.download_stock_data(commodity_symbols)
        all_data.update(commodity_data)

        # 3. Crypto
        crypto_symbols = self.config["assets"]["crypto"]
        crypto_data = self.download_crypto_data(crypto_symbols)
        all_data.update(crypto_data)

        # Résumé
        logger.info("-" * 60)
        logger.info(f"COLLECTE TERMINÉE : {len(all_data)} actifs téléchargés")
        for symbol, df in all_data.items():
            if df is not None and not df.empty:
                logger.info(
                    f"  {symbol:12s} : {len(df):4d} bougies | "
                    f"{df.index[0].date()} → {df.index[-1].date()}"
                )
        logger.info("=" * 60)

        return all_data
