"""
LANCEUR LOCAL — SYSTÈME DE TRADING IA COMPLET
==============================================

Ce script lance l'entraînement complet du système IA sur votre ordinateur.

COMMENT L'UTILISER :
    python run_local.py

DURÉE ESTIMÉE :
    CPU (Mac/PC standard) : 15-30 minutes
    GPU (NVIDIA)          : 5-10 minutes

CE QUI VA SE PASSER :
    1. Téléchargement des données réelles (Yahoo Finance)
    2. Calcul de 50+ indicateurs techniques
    3. Détection du régime de marché (Bull/Bear/Range)
    4. Entraînement LSTM + Transformer + CNN
    5. Entraînement agent RL (PPO)
    6. Backtesting avec résultats
    7. Optimisation du portefeuille
    8. Graphiques de performance
"""

import sys
import os
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Vérification des dépendances ──────────────────────────────────────────────
def check_dependencies():
    """Vérifie que toutes les librairies nécessaires sont installées."""
    required = {
        "torch":             "PyTorch (réseau de neurones)",
        "numpy":             "NumPy (calcul numérique)",
        "pandas":            "Pandas (manipulation de données)",
        "sklearn":           "Scikit-learn (machine learning)",
        "yfinance":          "Yahoo Finance (données de marché)",
        "hmmlearn":          "HMM (détection de régime)",
        "stable_baselines3": "Stable-Baselines3 (agent RL)",
        "matplotlib":        "Matplotlib (graphiques)",
        "yaml":              "PyYAML (configuration)",
    }

    print("Vérification des dépendances...")
    missing = []
    for pkg, description in required.items():
        try:
            __import__(pkg)
            print(f"  ✓ {description}")
        except ImportError:
            print(f"  ✗ {description} — MANQUANT")
            missing.append(pkg)

    if missing:
        print(f"\nInstallation manquante. Lancez :")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)

    print()


# ── Paramètres locaux (modifiables) ───────────────────────────────────────────
# Pour aller plus vite, on entraîne sur 3 actifs représentatifs.
# Vous pouvez ajouter/retirer des symboles ici.
SYMBOLES = [
    "AAPL",     # Apple — action technologique
    "BTC-USD",  # Bitcoin — crypto-monnaie
    "GC=F",     # Or — matière première refuge
]

# Nombre d'époques d'entraînement
# 50 = rapide (~5 min/actif sur CPU)   → bon pour tester
# 100 = standard (~10 min/actif sur CPU) → meilleure qualité
EPOCHS = 50

# Timesteps pour l'agent RL
# 50 000 = rapide (~2 min)
# 200 000 = standard (~8 min)
RL_TIMESTEPS = 50_000


# ══════════════════════════════════════════════════════════════════════════════
def main():
    import numpy as np
    import pandas as pd
    import torch
    import yaml
    import matplotlib
    matplotlib.use("Agg")  # Pas d'affichage interactif — sauvegarde en fichier
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    check_dependencies()

    # ── Configuration ──────────────────────────────────────────────────────────
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Surcharger le nombre d'époques avec notre valeur locale
    config["prediction_model"]["training"]["epochs"] = EPOCHS

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 62)
    print("   SYSTÈME DE TRADING IA — ENTRAÎNEMENT LOCAL COMPLET")
    print("=" * 62)
    print(f"  Actifs      : {', '.join(SYMBOLES)}")
    print(f"  Époques     : {EPOCHS} par actif")
    print(f"  RL Steps    : {RL_TIMESTEPS:,}")
    print(f"  Dispositif  : {device.upper()}")
    if device == "cuda":
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
    print("=" * 62)
    print()

    # Créer les dossiers
    for d in ["data/cache", "data/raw", "data/processed", "models", "results", "logs"]:
        os.makedirs(d, exist_ok=True)

    # ── ÉTAPE 1 : Téléchargement des données ──────────────────────────────────
    print("ÉTAPE 1/7 — Téléchargement des données de marché")
    print("-" * 50)

    from src.data.collectors import MarketDataCollector
    from src.data.preprocessor import DataPreprocessor

    collector = MarketDataCollector(config)
    raw_data = {}

    for symbol in SYMBOLES:
        print(f"  {symbol}...", end="", flush=True)
        try:
            data = collector.download_stock_data([symbol])
            if symbol in data and not data[symbol].empty:
                raw_data[symbol] = data[symbol]
                print(f" {len(data[symbol])} jours ✓")
            else:
                raise ValueError("Données vides")
        except Exception as e:
            # Fallback : données synthétiques réalistes
            print(f" connexion impossible → données synthétiques")
            raw_data[symbol] = _generate_synthetic_ohlcv(symbol)

    preprocessor = DataPreprocessor(config)
    processed_data = preprocessor.process(raw_data, fit=True)
    print(f"\n  → {len(processed_data)} actifs prêts\n")

    # ── ÉTAPE 2 : Features + Régimes ──────────────────────────────────────────
    print("ÉTAPE 2/7 — Calcul des indicateurs (50+ features)")
    print("-" * 50)

    from src.features.engineer import FeatureEngineer
    from src.models.regime_detector import MarketRegimeDetector

    fe = FeatureEngineer(config)
    features_data = {}

    for symbol, df in processed_data.items():
        print(f"  {symbol}...", end="", flush=True)
        df_feat = fe.engineer(df, symbol=symbol)
        features_data[symbol] = df_feat
        n_feat = len(fe.get_feature_columns(df_feat))
        print(f" {n_feat} features, {len(df_feat)} jours ✓")

    # Détection de régimes
    print("\n  Détection des régimes de marché (HMM)...")
    detector = MarketRegimeDetector(config)
    REGIME_NAMES = {0: "BULL", 1: "BEAR", 2: "RANGE", 3: "HIGH_VOL"}
    regime_results = {}

    for symbol, df in features_data.items():
        try:
            detector.fit(df)
            labels, probas = detector.predict(df)
            regime_results[symbol] = labels
            unique, counts = np.unique(labels, return_counts=True)
            regime_str = " | ".join(
                f"{REGIME_NAMES.get(int(u), str(u))}={c}" for u, c in zip(unique, counts)
            )
            print(f"  {symbol}: {regime_str}")
        except Exception as e:
            print(f"  {symbol}: régime non détecté ({e})")

    print()

    # ── ÉTAPE 3 : Entraînement du modèle ──────────────────────────────────────
    print("ÉTAPE 3/7 — Entraînement LSTM + Transformer + CNN")
    print("-" * 50)

    from src.models.prediction_model import HybridPredictionModel, ModelTrainer
    from src.data.pipeline import TimeSeriesDataset
    from torch.utils.data import DataLoader

    trained_models = {}
    training_histories = {}
    feat_cols_per_symbol = {}
    seq_len = config["features"]["sequence_length"]
    batch_size = config["prediction_model"]["training"]["batch_size"]

    for symbol in list(features_data.keys()):
        print(f"\n  [{symbol}]")
        df = features_data[symbol]
        feat_cols = fe.get_feature_columns(df)
        feat_cols_per_symbol[symbol] = feat_cols

        n = len(df)
        train_end = int(n * config["timeframes"]["train_ratio"])
        val_end   = int(n * (config["timeframes"]["train_ratio"] + config["timeframes"]["val_ratio"]))

        df_train = df.iloc[:train_end]
        df_val   = df.iloc[train_end:val_end]
        print(f"  Train: {len(df_train)} jours | Val: {len(df_val)} jours")

        def make_loader(df_slice, shuffle=False):
            X = df_slice[feat_cols].values.astype("float32")
            y = np.log(df_slice["close"] / df_slice["close"].shift(1)).fillna(0).values.astype("float32")
            ds = TimeSeriesDataset(X, y, seq_len)
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)

        try:
            train_loader = make_loader(df_train, shuffle=True)
            val_loader   = make_loader(df_val)

            model = HybridPredictionModel(n_features=len(feat_cols), config=config).to(device)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  Modèle : {n_params:,} paramètres")

            trainer = ModelTrainer(model, config, device)
            t0 = time.time()
            history = trainer.fit(train_loader, val_loader, model_path=f"models/{symbol}_model.pt")
            elapsed = time.time() - t0

            trained_models[symbol] = model
            training_histories[symbol] = history
            print(f"  Entraînement terminé en {elapsed:.0f}s ✓")
            print(f"  Modèle sauvegardé : models/{symbol}_model.pt")
        except Exception as e:
            print(f"  ERREUR : {e}")
            import traceback
            traceback.print_exc()

    print()

    # ── ÉTAPE 4 : Agent RL (PPO) ──────────────────────────────────────────────
    print("ÉTAPE 4/7 — Entraînement de l'agent RL (PPO)")
    print("-" * 50)

    from src.models.rl_agent import TradingEnvironment, RLAgent

    rl_agents = {}
    rl_symbols = list(features_data.keys())[:2]  # 2 actifs pour la vitesse

    for symbol in rl_symbols:
        print(f"\n  [{symbol}] {RL_TIMESTEPS:,} timesteps...")
        df = features_data[symbol]
        try:
            env = TradingEnvironment(df, config)
            agent = RLAgent(env, config)
            t0 = time.time()
            agent.train(total_timesteps=RL_TIMESTEPS)
            elapsed = time.time() - t0
            agent.save(f"models/{symbol}_rl_agent")
            rl_agents[symbol] = agent
            print(f"  Agent RL entraîné en {elapsed:.0f}s ✓")
        except Exception as e:
            print(f"  ERREUR RL : {e}")

    print()

    # ── ÉTAPE 5 : Backtesting ─────────────────────────────────────────────────
    print("ÉTAPE 5/7 — Backtesting (simulation sur données historiques)")
    print("-" * 50)

    from src.backtesting.backtester import Backtester
    import json

    backtester = Backtester(config)
    backtest_results = {}

    for symbol, model in trained_models.items():
        print(f"\n  [{symbol}]")
        df = features_data[symbol]
        feat_cols = feat_cols_per_symbol[symbol]
        n = len(df)
        val_end = int(n * (config["timeframes"]["train_ratio"] + config["timeframes"]["val_ratio"]))
        df_test = df.iloc[val_end:]

        # Générer les signaux
        model.eval()
        signals_list = []
        X_test = df_test[feat_cols].values.astype("float32")

        with torch.no_grad():
            for i in range(seq_len, len(X_test)):
                x_seq = torch.FloatTensor(X_test[i-seq_len:i]).unsqueeze(0).to(device)
                prob = float(model.predict_proba(x_seq).cpu())
                sig = 1 if prob > 0.52 else (-1 if prob < 0.48 else 0)
                signals_list.append(sig)

        signal_index = df_test.index[seq_len:]
        signals = pd.Series(signals_list, index=signal_index)

        n_buy  = (signals == 1).sum()
        n_sell = (signals == -1).sum()
        n_hold = (signals == 0).sum()
        print(f"  Signaux → BUY:{n_buy} | SELL:{n_sell} | HOLD:{n_hold}")

        try:
            result = backtester.run(df_test, signals, symbol=symbol)
            backtest_results[symbol] = result
            backtester.save_results(result, path=f"results/{symbol}_backtest.json")

            print(f"  Rendement total  : {result.total_return * 100:+.2f}%")
            print(f"  Sharpe ratio     : {result.sharpe_ratio:.3f}")
            print(f"  Max drawdown     : {result.max_drawdown * 100:.2f}%")
            print(f"  Trades           : {result.n_trades}")
            if result.n_trades > 0:
                print(f"  Win rate         : {result.win_rate * 100:.1f}%")
        except Exception as e:
            print(f"  ERREUR backtest : {e}")

    print()

    # ── ÉTAPE 6 : Optimisation du portefeuille ────────────────────────────────
    print("ÉTAPE 6/7 — Optimisation du portefeuille")
    print("-" * 50)

    from src.portfolio.portfolio_manager import PortfolioManager

    pm = PortfolioManager(config)
    portfolio_results = {}
    portfolio_df = {s: features_data[s] for s in features_data}

    for method in ["max_sharpe", "min_volatility", "risk_parity"]:
        try:
            weights = pm.optimize(portfolio_df, method=method)
            portfolio_results[method] = weights
            label = method.upper().replace("_", " ")
            print(f"\n  [{label}]")
            for sym, w in sorted(weights.items(), key=lambda x: -x[1]):
                bar = "█" * int(w * 30)
                print(f"  {sym:10s} {w * 100:5.1f}% {bar}")
        except Exception as e:
            print(f"  ERREUR {method} : {e}")

    print()

    # ── ÉTAPE 7 : Graphiques ──────────────────────────────────────────────────
    print("ÉTAPE 7/7 — Création du tableau de bord")
    print("-" * 50)

    try:
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle("SYSTÈME DE TRADING IA — TABLEAU DE BORD", fontsize=15, fontweight="bold")
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

        # ── Graphe 1 : Courbes de loss ──
        ax1 = fig.add_subplot(gs[0, 0])
        for sym, h in training_histories.items():
            if "train_loss" in h:
                ax1.plot(h["train_loss"], label=f"{sym} train", linewidth=1.5)
            if "val_loss" in h:
                ax1.plot(h["val_loss"], "--", label=f"{sym} val", linewidth=1)
        ax1.set_title("Perte d'entraînement", fontsize=10)
        ax1.set_xlabel("Époque")
        ax1.set_ylabel("Loss")
        ax1.legend(fontsize=7)
        ax1.grid(True, alpha=0.3)

        # ── Graphe 2 : Courbes de capital (backtest) ──
        ax2 = fig.add_subplot(gs[0, 1:])
        drawn = False
        for sym, result in backtest_results.items():
            if hasattr(result, "equity_curve") and result.equity_curve is not None and len(result.equity_curve) > 1:
                ax2.plot(result.equity_curve, label=sym, linewidth=1.5)
                drawn = True
        if not drawn:
            ax2.text(0.5, 0.5, "Pas assez de trades pour la courbe\n(normal avec données synthétiques)",
                     ha="center", va="center", transform=ax2.transAxes, fontsize=9, color="gray")
        ax2.axhline(y=100000, color="gray", linestyle="--", alpha=0.5, label="Capital initial")
        ax2.set_title("Évolution du capital (Backtest)", fontsize=10)
        ax2.set_ylabel("Valeur ($)")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # ── Graphe 3 : Allocation Max Sharpe ──
        ax3 = fig.add_subplot(gs[1, 0])
        if "max_sharpe" in portfolio_results:
            w = portfolio_results["max_sharpe"]
            labels = list(w.keys())
            vals   = [w[s] * 100 for s in labels]
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
            ax3.pie(vals, labels=labels, colors=colors, autopct="%1.0f%%", startangle=90)
        ax3.set_title("Allocation Max Sharpe", fontsize=10)

        # ── Graphe 4 : Allocation Min Volatilité ──
        ax4 = fig.add_subplot(gs[1, 1])
        if "min_volatility" in portfolio_results:
            w = portfolio_results["min_volatility"]
            labels = list(w.keys())
            vals   = [w[s] * 100 for s in labels]
            colors = plt.cm.Pastel1(np.linspace(0, 1, len(labels)))
            ax4.pie(vals, labels=labels, colors=colors, autopct="%1.0f%%", startangle=90)
        ax4.set_title("Allocation Min Volatilité", fontsize=10)

        # ── Graphe 5 : Rendements comparatifs ──
        ax5 = fig.add_subplot(gs[1, 2])
        if backtest_results:
            syms = list(backtest_results.keys())
            returns = [backtest_results[s].total_return * 100 for s in syms]
            sharpes = [backtest_results[s].sharpe_ratio for s in syms]
            x = np.arange(len(syms))
            width = 0.35
            bar_colors = ["#4CAF50" if r >= 0 else "#F44336" for r in returns]
            bars = ax5.bar(x - width/2, returns, width, label="Rendement %", color=bar_colors, alpha=0.8)
            ax5_twin = ax5.twinx()
            ax5_twin.bar(x + width/2, sharpes, width, label="Sharpe", color="#2196F3", alpha=0.6)
            ax5.set_xticks(x)
            ax5.set_xticklabels(syms, rotation=15, fontsize=8)
            ax5.set_ylabel("Rendement (%)")
            ax5_twin.set_ylabel("Sharpe")
            ax5.axhline(0, color="black", linewidth=0.8)
            ax5.set_title("Rendement & Sharpe", fontsize=10)
            ax5.grid(True, alpha=0.3, axis="y")
        else:
            ax5.text(0.5, 0.5, "Aucun résultat", ha="center", va="center", transform=ax5.transAxes)
            ax5.set_title("Rendements", fontsize=10)

        plt.savefig("results/tableau_de_bord.png", dpi=150, bbox_inches="tight")
        print("  Graphique sauvegardé : results/tableau_de_bord.png ✓")
    except Exception as e:
        print(f"  ERREUR graphique : {e}")

    # ── RÉSUMÉ FINAL ──────────────────────────────────────────────────────────
    print()
    print("=" * 62)
    print("   RÉSUMÉ FINAL")
    print("=" * 62)

    col_a = "ACTIF"
    col_r = "RENDEMENT"
    col_s = "SHARPE"
    col_d = "MAX DD"
    col_t = "TRADES"
    col_w = "WIN%"
    print(f"{col_a:12s} {col_r:>12s} {col_s:>8s} {col_d:>10s} {col_t:>8s} {col_w:>8s}")
    print("-" * 62)

    if backtest_results:
        for sym, res in backtest_results.items():
            ret  = f"{res.total_return * 100:+.2f}%"
            sh   = f"{res.sharpe_ratio:.3f}"
            dd   = f"{res.max_drawdown * 100:.2f}%"
            tr   = str(res.n_trades)
            wr   = f"{res.win_rate * 100:.1f}%" if res.n_trades > 0 else "N/A"
            print(f"{sym:12s} {ret:>12s} {sh:>8s} {dd:>10s} {tr:>8s} {wr:>8s}")
    else:
        print("  Aucun résultat de backtest disponible.")

    print("=" * 62)
    print()
    print("Modèles entraînés :")
    for f in sorted(os.listdir("models")) if os.path.exists("models") else []:
        size_kb = os.path.getsize(f"models/{f}") // 1024
        print(f"  models/{f}  ({size_kb} KB)")

    print()
    print("Fichiers de résultats :")
    for f in sorted(os.listdir("results")) if os.path.exists("results") else []:
        print(f"  results/{f}")

    print()
    print("Entraînement local terminé avec succès ✓")
    print()
    print("Pour changer les paramètres (actifs, époques) :")
    print("  → Modifiez SYMBOLES / EPOCHS / RL_TIMESTEPS en haut de run_local.py")
    print()


# ── Générateur de données synthétiques (fallback si pas d'internet) ────────────
def _generate_synthetic_ohlcv(symbol: str, n_days: int = 800) -> "pd.DataFrame":
    import numpy as np
    import pandas as pd

    params = {
        "AAPL":    (150.0,  0.00032, 0.013),
        "MSFT":    (280.0,  0.00035, 0.014),
        "GOOGL":   (130.0,  0.00028, 0.015),
        "NVDA":    (400.0,  0.00060, 0.025),
        "SPY":     (400.0,  0.00025, 0.010),
        "BTC-USD": (30000,  0.00150, 0.040),
        "ETH-USD": (2000,   0.00120, 0.045),
        "GC=F":    (1800.0, 0.00010, 0.007),
        "CL=F":    (75.0,   0.00005, 0.020),
    }
    start_price, drift, vol = params.get(symbol, (100.0, 0.00025, 0.015))

    rng   = np.random.default_rng(42)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days)
    rets  = rng.normal(drift, vol, n_days)
    close = start_price * np.exp(np.cumsum(rets))

    noise  = np.abs(rng.normal(0, vol * 0.4, n_days))
    high   = close * (1 + noise)
    low    = close * (1 - noise)
    open_  = np.roll(close, 1)
    open_[0] = close[0]
    volume = np.abs(rng.normal(5_000_000, 1_500_000, n_days)).astype(int)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


if __name__ == "__main__":
    main()
