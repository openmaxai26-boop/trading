"""
SYSTÈME DE TRADING IA — POINT D'ENTRÉE PRINCIPAL
==================================================

COMMENT LANCER LE SYSTÈME :

    # Mode démo (génère des signaux sans données réelles)
    python main.py --mode demo

    # Mode backtesting (télécharge les données et simule)
    python main.py --mode backtest --symbol AAPL

    # Mode production (paper trading - sans argent réel)
    python main.py --mode paper --symbol AAPL

    # Analyse complète de tous les actifs configurés
    python main.py --mode full

PRÉREQUIS :
    pip install -r requirements.txt

STRUCTURE DU PROJET :
    config/config.yaml          ← Modifiez ici les paramètres
    src/data/                   ← Collecte et nettoyage des données
    src/features/               ← Calcul des indicateurs
    src/models/                 ← LSTM+Transformer+CNN, HMM, RL
    src/risk/                   ← Gestion des risques
    src/portfolio/              ← Allocation du capital
    src/backtesting/            ← Simulation historique
    src/continuous_learning/    ← Réentraînement automatique
"""

import argparse
import yaml
import sys
from pathlib import Path

# ── Ajouter le répertoire racine au PATH ──
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import get_logger

logger = get_logger("Main")


def load_config(config_path: str = "./config/config.yaml") -> dict:
    """
    Charge la configuration depuis le fichier YAML.

    LE FICHIER YAML :
    C'est un format de configuration lisible par les humains.
    Exemple :
        system:
          mode: "paper"
          log_level: "INFO"

    PARAMÈTRE :
    - config_path : Chemin vers le fichier de configuration

    RETOURNE :
    - Dictionnaire Python avec toute la configuration
    """
    path = Path(config_path)
    if not path.exists():
        logger.error(f"Fichier de configuration introuvable : {config_path}")
        logger.error("Créez ou vérifiez le fichier config/config.yaml")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"Configuration chargée depuis : {config_path}")
    return config


def run_demo(config: dict):
    """
    Mode DÉMO : Démontre les capacités du système sans données réelles.

    UTILITÉ :
    - Vérifier que tout le code fonctionne
    - Comprendre les sorties du système
    - Tester sans connexion internet
    """
    import numpy as np
    import pandas as pd
    import torch

    logger.info("=" * 60)
    logger.info("MODE DÉMO — Génération de données synthétiques")
    logger.info("=" * 60)

    # ── 1. Générer des données OHLCV synthétiques ──
    np.random.seed(config["system"]["random_seed"])
    n_days = 500
    dates  = pd.date_range("2022-01-01", periods=n_days, freq="B")

    # Simuler un chemin aléatoire (Geometric Brownian Motion)
    # C'est le modèle de base du prix des actions (Black-Scholes)
    drift   = 0.0003   # Tendance journalière (~7.5% annuel)
    vol     = 0.015    # Volatilité journalière (~24% annuel)
    returns = np.random.normal(drift, vol, n_days)
    prices  = 100 * np.exp(np.cumsum(returns))

    # Construire OHLCV réaliste
    high = prices * (1 + np.abs(np.random.normal(0, 0.005, n_days)))
    low  = prices * (1 - np.abs(np.random.normal(0, 0.005, n_days)))
    open_ = np.roll(prices, 1); open_[0] = prices[0]
    vol_data = np.abs(np.random.normal(1_000_000, 300_000, n_days)).astype(int)

    df = pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": prices, "volume": vol_data
    }, index=dates)

    logger.info(f"Données synthétiques créées : {n_days} jours")
    logger.info(f"Prix initial : {prices[0]:.2f}$ | Prix final : {prices[-1]:.2f}$")

    # ── 2. Feature Engineering ──
    logger.info("\n--- FEATURE ENGINEERING ---")
    from src.features.engineer import FeatureEngineer
    fe = FeatureEngineer(config)
    df_feat = fe.engineer(df, symbol="DEMO")
    feat_cols = fe.get_feature_columns(df_feat)
    logger.info(f"Features générées : {len(feat_cols)} colonnes")

    # ── 3. Détecteur de régimes ──
    logger.info("\n--- DÉTECTION DE RÉGIMES ---")
    from src.models.regime_detector import MarketRegimeDetector, REGIME_NAMES
    detector = MarketRegimeDetector(config)
    detector.fit(df_feat)
    regime_info = detector.predict_current(df_feat)
    logger.info(f"Régime actuel       : {regime_info['regime_name']}")
    logger.info(f"Confiance           : {regime_info['confidence']:.0%}")
    logger.info(f"Conseil             : {regime_info['strategy_hint']}")

    # ── 4. Modèle de prédiction ──
    logger.info("\n--- MODÈLE DE PRÉDICTION (LSTM+Transformer+CNN) ---")
    from src.models.prediction_model import HybridPredictionModel
    n_features = len(feat_cols)
    model = HybridPredictionModel(n_features=n_features, config=config)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Modèle créé : {n_params:,} paramètres")

    # Simulation d'un signal (sans entraînement réel en mode démo)
    seq_len = config["features"]["sequence_length"]
    X_demo  = df_feat[feat_cols].values[-seq_len:]
    X_tensor = torch.FloatTensor(X_demo).unsqueeze(0)  # Ajouter dimension batch

    with torch.no_grad():
        signal_info = model.get_signal(X_tensor)

    logger.info(f"Signal généré       : {signal_info['signal']}")
    logger.info(f"Probabilité hausse  : {signal_info['probability_up']}%")
    logger.info(f"Score de confiance  : {signal_info['confidence_score']}%")
    logger.info(f"Niveau de risque    : {signal_info['risk_level']}")
    logger.info(f"Rendement attendu   : {signal_info['expected_return']:+.3f}%")
    logger.info(f"Incertitude         : {signal_info['uncertainty']:.3f}%")

    # ── 5. Risk Manager ──
    logger.info("\n--- GESTION DES RISQUES ---")
    from src.risk.risk_manager import RiskManager
    rm = RiskManager(config)
    allowed, size, reason = rm.check_trade(
        symbol="DEMO",
        signal=signal_info["signal"],
        portfolio_value=100_000,
        current_positions={},
        volatility=0.20,
        regime=regime_info["regime_name"]
    )
    logger.info(f"Trade autorisé      : {allowed}")
    logger.info(f"Taille recommandée  : {size:.1%}")
    logger.info(f"Raison              : {reason}")

    # ── 6. Portfolio Manager ──
    logger.info("\n--- ALLOCATION DE PORTEFEUILLE ---")
    from src.portfolio.portfolio_manager import PortfolioManager
    pm = PortfolioManager(config)

    # Créer un DataFrame multi-actif simplifié
    multi_prices = pd.DataFrame({
        "AAPL" : 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, n_days))),
        "BTC"  : 100 * np.exp(np.cumsum(np.random.normal(0.0002, 0.030, n_days))),
        "GC=F" : 100 * np.exp(np.cumsum(np.random.normal(0.0001, 0.008, n_days))),
    }, index=dates)

    weights = pm.optimize(multi_prices, method="max_sharpe")
    logger.info("Poids optimaux (Max Sharpe) :")
    for sym, w in weights.items():
        logger.info(f"  {sym:<10} : {w:.1%}")

    # ── 7. Résumé final ──
    logger.info("\n" + "=" * 60)
    logger.info("RÉSUMÉ DU MODE DÉMO")
    logger.info("=" * 60)
    logger.info("✓ Données synthétiques générées (500 jours)")
    logger.info(f"✓ {len(feat_cols)} features calculées")
    logger.info(f"✓ Régime détecté : {regime_info['regime_name']}")
    logger.info(f"✓ Signal : {signal_info['signal']} ({signal_info['probability_up']}% probabilité)")
    logger.info(f"✓ Risque : {signal_info['risk_level']}")
    logger.info(f"✓ Allocation optimale : {len(weights)} actifs")
    logger.info("=" * 60)
    logger.info("Pour un backtest complet, lancez : python main.py --mode backtest")
    logger.info("=" * 60)


def _generate_synthetic_ohlcv(
    symbol: str,
    n_days: int = 800,
    seed: int = 42
) -> "pd.DataFrame":
    """
    Génère des données OHLCV synthétiques réalistes (Geometric Brownian Motion).
    Utilisé quand Yahoo Finance est inaccessible (pas d'internet).

    Les paramètres drift/vol sont calibrés sur des actifs réels typiques :
    - Actions US : drift ~8%/an, vol ~20%/an
    - Bitcoin    : drift ~50%/an, vol ~70%/an
    - Or         : drift ~5%/an, vol ~12%/an
    """
    import numpy as np
    import pandas as pd

    params = {
        "AAPL" : (150.0, 0.00032, 0.013), "MSFT"  : (280.0, 0.00035, 0.014),
        "GOOGL": (130.0, 0.00028, 0.015), "AMZN"  : (160.0, 0.00030, 0.018),
        "NVDA" : (400.0, 0.00060, 0.025), "SPY"   : (400.0, 0.00025, 0.010),
        "BTC-USD": (30000, 0.00150, 0.040), "ETH-USD": (2000, 0.00120, 0.045),
        "BNB-USD": (250.0, 0.00100, 0.038), "GC=F" : (1800.0, 0.00010, 0.007),
        "CL=F" : (75.0, 0.00005, 0.020),  "SI=F"  : (22.0, 0.00008, 0.012),
    }
    start_price, drift, vol = params.get(symbol, (100.0, 0.00025, 0.015))

    rng  = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days)
    rets  = rng.normal(drift, vol, n_days)
    close = start_price * np.exp(np.cumsum(rets))

    noise  = np.abs(rng.normal(0, vol * 0.4, n_days))
    high   = close * (1 + noise)
    low    = close * (1 - noise)
    open_  = np.roll(close, 1); open_[0] = close[0]
    volume = np.abs(rng.normal(5_000_000, 1_500_000, n_days)).astype(int)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates
    )


def run_backtest(config: dict, symbol: str = "AAPL"):
    """
    Mode BACKTESTING : télécharge les données réelles et simule la stratégie.

    ÉTAPES :
    1. Télécharger les données OHLCV (Yahoo Finance)
    2. Calculer les features
    3. Entraîner le modèle de prédiction
    4. Générer les signaux sur la période de test
    5. Simuler les trades avec frais et slippage
    6. Afficher le rapport de performance
    """
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader

    logger.info("=" * 60)
    logger.info(f"MODE BACKTESTING — {symbol}")
    logger.info("=" * 60)

    # ── 1. Collecte des données ──
    from src.data.collectors import MarketDataCollector
    from src.data.preprocessor import DataPreprocessor

    collector = MarketDataCollector(config)
    raw_data = collector.download_stock_data([symbol])

    if symbol not in raw_data:
        logger.warning(f"Téléchargement impossible pour {symbol} → génération de données synthétiques réalistes")
        raw_data = {symbol: _generate_synthetic_ohlcv(symbol, n_days=800, seed=42)}

    df_raw = raw_data[symbol]

    # ── 2. Prétraitement ──
    preprocessor = DataPreprocessor(config)
    df_clean = preprocessor.process({symbol: df_raw})[symbol]

    # ── 3. Feature Engineering ──
    from src.features.engineer import FeatureEngineer
    fe = FeatureEngineer(config)
    df_feat = fe.engineer(df_clean, symbol=symbol)
    feat_cols = fe.get_feature_columns(df_feat)

    # ── 4. Division train/test ──
    n = len(df_feat)
    train_end = int(n * config["timeframes"]["train_ratio"])
    val_end   = int(n * (config["timeframes"]["train_ratio"] + config["timeframes"]["val_ratio"]))

    df_train = df_feat.iloc[:train_end]
    df_val   = df_feat.iloc[train_end:val_end]
    df_test  = df_feat.iloc[val_end:]

    logger.info(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)} jours")

    # ── 5. Créer les séquences ──
    from src.data.pipeline import TimeSeriesDataset

    seq_len = config["features"]["sequence_length"]
    batch   = config["prediction_model"]["training"]["batch_size"]

    def make_loader(df_slice, shuffle=False):
        X = df_slice[feat_cols].values.astype("float32")
        y = np.log(df_slice["close"] / df_slice["close"].shift(1)).fillna(0).values.astype("float32")
        ds = TimeSeriesDataset(X, y, seq_len)
        return DataLoader(ds, batch_size=batch, shuffle=shuffle)

    train_loader = make_loader(df_train, shuffle=True)
    val_loader   = make_loader(df_val)
    test_loader  = make_loader(df_test)

    # ── 6. Entraîner le modèle ──
    from src.models.prediction_model import HybridPredictionModel, ModelTrainer

    model   = HybridPredictionModel(n_features=len(feat_cols), config=config)
    trainer = ModelTrainer(model, config)
    trainer.fit(train_loader, val_loader, model_path=f"./models/{symbol}_model.pt")
    eval_results = trainer.evaluate(test_loader)

    # ── 7. Générer les signaux sur les données de test ──
    model.eval()
    signals_list = []

    device = next(model.parameters()).device
    probs_all = []
    with torch.no_grad():
        X_test = df_test[feat_cols].values.astype("float32")
        for i in range(seq_len, len(X_test)):
            x_seq = torch.FloatTensor(X_test[i-seq_len:i]).unsqueeze(0).to(device)
            prob = float(model.predict_proba(x_seq).cpu())
            probs_all.append(prob)
            sig = 1 if prob > 0.52 else (-1 if prob < 0.48 else 0)
            signals_list.append(sig)

    if probs_all:
        import numpy as _np
        logger.info(
            f"Distribution des probabilités : "
            f"min={_np.min(probs_all):.2f} | moy={_np.mean(probs_all):.2f} | max={_np.max(probs_all):.2f} | "
            f"BUY={sum(p>0.52 for p in probs_all)} | SELL={sum(p<0.48 for p in probs_all)} | HOLD={sum(0.48<=p<=0.52 for p in probs_all)}"
        )

    # Aligner les signaux avec les dates de test
    signal_index = df_test.index[seq_len:]
    signals = pd.Series(signals_list, index=signal_index)

    # ── 8. Backtesting ──
    from src.backtesting.backtester import Backtester
    bt = Backtester(config)
    result = bt.run(df_test, signals, symbol=symbol)

    # ── 9. Sauvegarder ──
    bt.save_results(result, path=f"./results/{symbol}_backtest.json")
    return result


def run_full_analysis(config: dict):
    """
    Analyse complète de tous les actifs configurés.
    Lance le backtesting pour chaque actif et génère un rapport global.
    """
    all_symbols = (
        config["assets"]["stocks"] +
        config["assets"]["crypto"] +
        config["assets"]["commodities"]
    )

    logger.info(f"Analyse complète : {len(all_symbols)} actifs")
    results = {}

    for symbol in all_symbols:
        logger.info(f"\n{'─' * 50}")
        logger.info(f"Analyse : {symbol}")
        try:
            result = run_backtest(config, symbol=symbol)
            if result:
                results[symbol] = result.to_dict()
        except Exception as e:
            logger.error(f"{symbol} : Erreur → {e}")

    # Rapport comparatif
    if results:
        logger.info("\n" + "=" * 65)
        logger.info(f"{'ACTIF':<12} {'Sharpe':>8} {'Rendement':>12} {'MaxDD':>10} {'WinRate':>10}")
        logger.info("=" * 65)
        for sym, r in results.items():
            logger.info(
                f"{sym:<12} {str(r.get('sharpe_ratio','?')):>8} "
                f"{str(r.get('rendement_total','?')):>12} "
                f"{str(r.get('max_drawdown','?')):>10} "
                f"{str(r.get('win_rate','?')):>10}"
            )
        logger.info("=" * 65)


def main():
    """Point d'entrée principal avec parseur d'arguments."""
    parser = argparse.ArgumentParser(
        description="Système de Trading IA — Architecture institutionnelle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXEMPLES D'UTILISATION :
  python main.py --mode demo
  python main.py --mode backtest --symbol AAPL
  python main.py --mode backtest --symbol BTC-USD
  python main.py --mode full
  python main.py --config config/config.yaml --mode demo
        """
    )
    parser.add_argument(
        "--mode", choices=["demo", "backtest", "full"],
        default="demo",
        help="Mode d'exécution (défaut: demo)"
    )
    parser.add_argument(
        "--symbol", type=str, default="AAPL",
        help="Symbole de l'actif pour le backtesting (défaut: AAPL)"
    )
    parser.add_argument(
        "--config", type=str, default="./config/config.yaml",
        help="Chemin vers le fichier de configuration"
    )

    args = parser.parse_args()

    # Charger la configuration
    config = load_config(args.config)

    logger.info("╔══════════════════════════════════════════════════════╗")
    logger.info("║      SYSTÈME DE TRADING IA — VERSION 1.0             ║")
    logger.info("║      Architecture institutionnelle complète           ║")
    logger.info("╚══════════════════════════════════════════════════════╝")
    logger.info(f"Mode : {args.mode.upper()}")
    logger.info(f"Système : {config['system']['mode'].upper()}")

    # Créer les dossiers nécessaires
    for d in ["./data/cache", "./data/raw", "./data/processed",
              "./models", "./results", "./logs"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Lancer le mode choisi
    if args.mode == "demo":
        run_demo(config)

    elif args.mode == "backtest":
        run_backtest(config, symbol=args.symbol)

    elif args.mode == "full":
        run_full_analysis(config)


if __name__ == "__main__":
    main()
