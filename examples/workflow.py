"""
WORKFLOW COMPLET — DE LA DONNÉE BRUTE À LA PRÉDICTION
=======================================================

Ce fichier est un TUTORIEL pas à pas.
Chaque étape est expliquée et vous pouvez l'exécuter
bloc par bloc pour comprendre ce qui se passe.

COMMENT L'EXÉCUTER :
    cd /chemin/vers/trading
    python examples/workflow.py

POUR LES DÉBUTANTS ABSOLUS :
Ce fichier montre exactement comment les 9 couches
du système travaillent ensemble. Lisez les commentaires
et exécutez le code étape par étape.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml
import torch

# ══════════════════════════════════════════════════════════════════
#  ÉTAPE 0 : Charger la configuration
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("ÉTAPE 0 : Chargement de la configuration")
print("=" * 65)

with open("./config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

print(f"✓ Système     : {config['system']['name']} v{config['system']['version']}")
print(f"✓ Mode        : {config['system']['mode']}")
print(f"✓ Actifs stock: {config['assets']['stocks']}")
print(f"✓ Capital init: {config['portfolio']['initial_capital']:,}$")

# ══════════════════════════════════════════════════════════════════
#  ÉTAPE 1 : Générer des données OHLCV synthétiques (pour le démo)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("ÉTAPE 1 : Création de données de marché synthétiques")
print("=" * 65)
print("""
QU'EST-CE QUE OHLCV ?
  O = Open  → Prix d'ouverture de la bougie (journée/heure)
  H = High  → Prix le plus haut de la période
  L = Low   → Prix le plus bas de la période
  C = Close → Prix de clôture de la période
  V = Volume → Nombre d'unités échangées

Ces 5 colonnes sont les DONNÉES DE BASE de tout système de trading.
On génère ici des données synthétiques pour le tutoriel.
""")

np.random.seed(42)
n_days = 600
dates  = pd.date_range("2022-01-01", periods=n_days, freq="B")  # Jours ouvrables

# Geometric Brownian Motion : modèle standard du mouvement des prix
drift   = 0.0003  # Tendance haussière légère
vol_day = 0.015   # Volatilité journalière = 15%/√252 ≈ 24% annuel
returns = np.random.normal(drift, vol_day, n_days)
close   = 150.0 * np.exp(np.cumsum(returns))  # Prix de départ : 150$

# Construire OHLCV réaliste
noise   = np.abs(np.random.normal(0, 0.003, n_days))
high    = close * (1 + noise + np.abs(np.random.normal(0, 0.002, n_days)))
low     = close * (1 - noise - np.abs(np.random.normal(0, 0.002, n_days)))
open_   = np.roll(close, 1); open_[0] = close[0]
volume  = np.abs(np.random.normal(5_000_000, 1_500_000, n_days)).astype(int)

df = pd.DataFrame({
    "open": open_, "high": high, "low": low,
    "close": close, "volume": volume
}, index=dates)

print(f"✓ {n_days} jours de données créées")
print(f"✓ Prix initial : {close[0]:.2f}$")
print(f"✓ Prix final   : {close[-1]:.2f}$")
print(f"✓ Rendement brut : {(close[-1]/close[0]-1)*100:+.1f}%")
print(f"\nAperçu des données :")
print(df.tail(3).round(2).to_string())

# ══════════════════════════════════════════════════════════════════
#  ÉTAPE 2 : Feature Engineering
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("ÉTAPE 2 : Ingénierie des features (variables)")
print("=" * 65)
print("""
POURQUOI DES FEATURES ?
Le modèle ne peut pas apprendre directement des prix bruts.
On calcule des INDICATEURS qui résument l'information :

• RSI           → Le marché est-il en surachat/survente ?
• MACD          → La tendance est-elle haussière/baissière ?
• Bollinger     → Le prix est-il à un extrême ?
• Volatilité    → Le risque est-il élevé en ce moment ?
• ...et 50+ autres features
""")

from src.features.engineer import FeatureEngineer
fe = FeatureEngineer(config)
df_feat = fe.engineer(df, symbol="EXEMPLE")
feat_cols = fe.get_feature_columns(df_feat)

print(f"✓ {len(feat_cols)} features calculées")
print(f"\nListe des features :")
for i, col in enumerate(feat_cols[:15], 1):
    val = df_feat[col].iloc[-1]
    print(f"  {i:2d}. {col:<35}: {val:+.4f}")
print(f"  ... et {len(feat_cols)-15} autres features")

# ══════════════════════════════════════════════════════════════════
#  ÉTAPE 3 : Détection du régime de marché
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("ÉTAPE 3 : Détection du régime de marché (HMM)")
print("=" * 65)
print("""
QU'EST-CE QU'UN RÉGIME DE MARCHÉ ?
Le marché change de "personnalité" au cours du temps :

  BULL     → Tendance haussière, faible peur
  BEAR     → Tendance baissière, stress élevé
  RANGE    → Marché latéral, oscillations
  HIGH_VOL → Extrême volatilité, incertitude maximale

Le modèle Hidden Markov Model (HMM) détecte automatiquement
dans quel régime nous sommes à chaque instant.
""")

from src.models.regime_detector import MarketRegimeDetector, REGIME_NAMES
detector = MarketRegimeDetector(config)
detector.fit(df_feat)

regimes, probas = detector.predict(df_feat)
current = detector.predict_current(df_feat)

print(f"✓ Régime ACTUEL     : {current['regime_name']}")
print(f"✓ Confiance         : {current['confidence']:.0%}")
print(f"✓ Probabilités      :")
for name, p in current['probabilities'].items():
    bar = "█" * int(p * 20)
    print(f"     {name:<10}: {p:.0%} {bar}")
print(f"\n✓ Conseil stratégique : {current['strategy_hint']}")

# Distribution des régimes sur toute la période
from collections import Counter
regime_dist = Counter(regimes)
print(f"\n✓ Distribution historique des régimes :")
for r_id, count in sorted(regime_dist.items()):
    pct = count / len(regimes) * 100
    bar = "█" * int(pct / 2)
    print(f"     {REGIME_NAMES[r_id]:<10}: {pct:.0f}% {bar}")

# ══════════════════════════════════════════════════════════════════
#  ÉTAPE 4 : Modèle de prédiction hybride
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("ÉTAPE 4 : Modèle de prédiction LSTM + Transformer + CNN")
print("=" * 65)
print("""
ARCHITECTURE HYBRIDE :

  ┌──────────────────────────────────────────────────────┐
  │          Input : 60 derniers jours × N features      │
  └──────────────┬──────────────┬─────────────┬──────────┘
                 │              │             │
              [LSTM]    [Transformer]       [CNN]
          Mémoire long    Attention        Patterns
              terme       globale         locaux
                 │              │             │
                 └──────── Fusion ───────────┘
                                │
                        [Couches denses]
                                │
                   ┌────────────┴────────────┐
                   │                         │
                 [mu]                   [log_var]
             Rendement               Incertitude
               prédit                  prédite
                   │                         │
                   └────── Distribution ─────┘
                                │
                     Probabilité de hausse
""")

from src.models.prediction_model import HybridPredictionModel
n_feat  = len(feat_cols)
model   = HybridPredictionModel(n_features=n_feat, config=config)

n_params = sum(p.numel() for p in model.parameters())
print(f"✓ Modèle créé avec {n_params:,} paramètres")
print(f"✓ Architecture : LSTM({config['prediction_model']['lstm']['hidden_size']}) + "
      f"Transformer(d={config['prediction_model']['transformer']['d_model']}, "
      f"heads={config['prediction_model']['transformer']['nhead']}) + "
      f"CNN{config['prediction_model']['cnn']['channels']}")

# NOTE : On ne fait pas un vrai entraînement ici (trop long pour un exemple).
# On montre juste les sorties du modèle non entraîné.
print("\n(Note : Le modèle n'est pas entraîné dans cet exemple)")
print("  → Pour entraîner : python main.py --mode backtest --symbol AAPL")

seq_len  = config["features"]["sequence_length"]
X_sample = df_feat[feat_cols].values[-seq_len:].astype("float32")
X_tensor = torch.FloatTensor(X_sample).unsqueeze(0)

model.eval()
with torch.no_grad():
    signal = model.get_signal(X_tensor)

print(f"\n✓ Signal du modèle (non entraîné — exemple) :")
print(f"     Signal              : {signal['signal']}")
print(f"     Probabilité hausse  : {signal['probability_up']}%")
print(f"     Score de confiance  : {signal['confidence_score']}%")
print(f"     Niveau de risque    : {signal['risk_level']}")
print(f"     Rendement attendu   : {signal['expected_return']:+.3f}%")
print(f"     Incertitude         : {signal['uncertainty']:.3f}%")

# ══════════════════════════════════════════════════════════════════
#  ÉTAPE 5 : Vérification des risques
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("ÉTAPE 5 : Vérification des risques")
print("=" * 65)
print("""
AVANT D'EXÉCUTER UN TRADE, LE SYSTÈME VÉRIFIE :
  1. Est-ce que le drawdown est dans les limites ?
  2. Est-ce que la perte journalière est acceptable ?
  3. Est-ce que la taille de la position est raisonnable ?
  4. Y a-t-il un circuit breaker actif ?
""")

from src.risk.risk_manager import RiskManager, RiskStatus

rm = RiskManager(config)

# Simuler des vérifications de risque dans différents scénarios
scenarios = [
    {"signal": signal["signal"], "vol": 0.15, "regime": current["regime_name"],
     "positions": {}, "desc": "Marché normal"},
    {"signal": "BUY", "vol": 0.50, "regime": "HIGH_VOL",
     "positions": {}, "desc": "Haute volatilité crypto"},
    {"signal": "BUY", "vol": 0.15, "regime": "BEAR",
     "positions": {"AAPL": 8000, "MSFT": 7000, "GOOGL": 6000, "AMZN": 5000,
                   "NVDA": 5000, "BTC": 4000, "ETH": 3000}, "desc": "Portfolio saturé"},
]

for s in scenarios:
    allowed, size, reason = rm.check_trade(
        symbol="TEST",
        signal=s["signal"],
        portfolio_value=100_000,
        current_positions=s["positions"],
        volatility=s["vol"],
        regime=s["regime"]
    )
    status_icon = "✓" if allowed else "✗"
    print(f"\n  Scénario : {s['desc']}")
    print(f"  {status_icon} Autorisé={allowed} | Taille={size:.1%}")
    print(f"    → {reason}")

# ══════════════════════════════════════════════════════════════════
#  ÉTAPE 6 : Simulation d'un backtest simplifié
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("ÉTAPE 6 : Backtest simplifié (stratégie momentum)")
print("=" * 65)
print("""
STRATÉGIE TESTÉE : Momentum simple
  Signal d'ACHAT  → si RSI < 40 (survente)
  Signal de VENTE → si RSI > 65 (surachat)
  Sinon HOLD      → ne rien faire

Cette stratégie est SIMPLE et sert d'EXEMPLE.
Le vrai système utilise le modèle LSTM+Transformer+CNN.
""")

from src.backtesting.backtester import Backtester

# Générer des signaux momentum simples sur la partie test
test_df = df_feat.iloc[int(n_days * 0.85):]  # 15% de test

# Signal basé sur RSI
rsi_vals = test_df["rsi"] if "rsi" in test_df.columns else pd.Series(50, index=test_df.index)
signals_momentum = pd.Series(0, index=test_df.index)
signals_momentum[rsi_vals < 40] = 1    # Achat en survente
signals_momentum[rsi_vals > 65] = -1   # Vente en surachat

bt = Backtester(config)
result = bt.run(test_df, signals_momentum, symbol="EXEMPLE_MOMENTUM")

# ══════════════════════════════════════════════════════════════════
#  ÉTAPE 7 : Portfolio multi-actifs
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("ÉTAPE 7 : Optimisation de portefeuille multi-actifs")
print("=" * 65)
print("""
THÉORIE DE MARKOWITZ (1952) :
En combinant des actifs avec des corrélations DIFFÉRENTES,
on peut obtenir le MÊME rendement avec MOINS de risque.

  Portefeuille 1 : 100% AAPL  → Rendement 15%, Risque 25%
  Portefeuille 2 : 50% AAPL + 50% Or → Rendement 12%, Risque 15%

→ On sacrifie 3% de rendement mais on réduit le risque de 40% !
""")

from src.portfolio.portfolio_manager import PortfolioManager
pm = PortfolioManager(config)

# Simuler 4 actifs avec des dynamiques différentes
np.random.seed(123)
n_sim = 400
sim_dates = pd.date_range("2023-01-01", periods=n_sim, freq="B")

multi_prices = pd.DataFrame({
    "Actions (AAPL)": 100 * np.exp(np.cumsum(np.random.normal(0.0004, 0.016, n_sim))),
    "Crypto (BTC)  ": 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.035, n_sim))),
    "Or (GC=F)     ": 100 * np.exp(np.cumsum(np.random.normal(0.0001, 0.007, n_sim))),
    "Obligations   ": 100 * np.exp(np.cumsum(np.random.normal(0.00015, 0.003, n_sim))),
}, index=sim_dates)

print("Méthode 1 : Maximum Sharpe Ratio")
w_sharpe = pm.optimize(multi_prices, method="max_sharpe")
print("  Poids :")
for sym, w in w_sharpe.items():
    bar = "█" * int(w * 25)
    print(f"    {sym}: {w:.1%} {bar}")

print("\nMéthode 2 : Risk Parity (chaque actif contribue également au risque)")
w_rp = pm.optimize(multi_prices, method="risk_parity")
print("  Poids :")
for sym, w in w_rp.items():
    bar = "█" * int(w * 25)
    print(f"    {sym}: {w:.1%} {bar}")

print("\nMéthode 3 : Équipondéré (référence simple)")
w_eq = pm.optimize(multi_prices, method="equal_weight")
print("  Poids :")
for sym, w in w_eq.items():
    bar = "█" * int(w * 25)
    print(f"    {sym}: {w:.1%} {bar}")

# ══════════════════════════════════════════════════════════════════
#  RÉSUMÉ FINAL
# ══════════════════════════════════════════════════════════════════
print("\n" + "╔" + "═" * 63 + "╗")
print("║" + " WORKFLOW COMPLET TERMINÉ ".center(63) + "║")
print("╚" + "═" * 63 + "╝")

print("""
CE QUE VOUS VENEZ DE VOIR :
  1. ✓ Données OHLCV créées (600 jours de données synthétiques)
  2. ✓ Features engineering (60+ indicateurs calculés)
  3. ✓ Détection de régime HMM (BULL/BEAR/RANGE/HIGH_VOL)
  4. ✓ Modèle LSTM+Transformer+CNN instancié
  5. ✓ Gestion des risques (3 scénarios testés)
  6. ✓ Backtesting avec métriques complètes
  7. ✓ Optimisation de portefeuille (3 méthodes)

PROCHAINES ÉTAPES :
  → Installer les dépendances  : pip install -r requirements.txt
  → Lancer le mode démo        : python main.py --mode demo
  → Lancer un backtest réel    : python main.py --mode backtest --symbol AAPL
  → Analyser tous les actifs   : python main.py --mode full

FICHIERS CLÉS À MODIFIER :
  → config/config.yaml         : Paramètres (capital, actifs, risques)
  → src/models/prediction_model.py : Architecture du réseau de neurones
  → src/risk/risk_manager.py   : Limites de risque

AVERTISSEMENT LÉGAL :
  Ce système est à des fins ÉDUCATIVES uniquement.
  Les performances passées ne garantissent pas les performances futures.
  Ne risquez jamais de l'argent que vous ne pouvez pas vous permettre de perdre.
""")
