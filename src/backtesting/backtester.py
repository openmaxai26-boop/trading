"""
MOTEUR DE BACKTESTING  (Couche 8)
===================================

QU'EST-CE QUE LE BACKTESTING ?
Le backtesting = tester une stratégie sur des données PASSÉES
pour estimer ses performances FUTURES.

ANALOGIE :
Avant de prendre le volant pour la première fois, vous
simulez sur un simulateur. Le backtesting est le simulateur
du trader : vous testez la stratégie sans risquer d'argent réel.

LIMITES IMPORTANTES (honnêteté absolue) :
1. Les performances passées NE GARANTISSENT PAS les performances futures
2. Le "look-ahead bias" peut fausser les résultats
   (utiliser des données du futur par inadvertance)
3. L'"overfitting" : une stratégie trop optimisée sur le passé
   échoue souvent en conditions réelles

RÉALISME DE LA SIMULATION :
Ce backtester intègre :
- Frais de transaction (0.1% par défaut)
- Slippage (0.05% de glissement de prix)
- Walk-forward validation (voir ci-dessous)

WALK-FORWARD VALIDATION :
Au lieu d'un seul backtesting, on fait une série glissante :
  Fenêtre 1 : Train [Jan-Déc 2022] → Test [Jan-Mar 2023]
  Fenêtre 2 : Train [Avr 2022-Mar 2023] → Test [Avr-Jun 2023]
  Fenêtre 3 : Train [Juil 2022-Jun 2023] → Test [Juil-Sep 2023]
  ...
C'est beaucoup plus robuste qu'un seul test statique.

MÉTRIQUES CALCULÉES :
- Sharpe Ratio    : Rendement ajusté au risque (objectif > 1.0)
- Sortino Ratio   : Comme Sharpe mais pénalise seulement les pertes
- Max Drawdown    : Perte maximale depuis un sommet (objectif < -15%)
- Win Rate        : % de trades gagnants (objectif > 50%)
- Profit Factor   : Gains totaux / Pertes totales (objectif > 1.5)
- CAGR            : Rendement annuel composé
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json

from ..utils.logger import get_logger

logger = get_logger("Backtester")


@dataclass
class BacktestResult:
    """
    Résultats complets d'un backtesting.

    Tous les chiffres sont calculés à partir de la simulation.
    Aucune valeur n'est inventée ou estimée.
    """
    # Identification
    symbol    : str
    start_date: str
    end_date  : str
    n_days    : int

    # Capital
    initial_capital: float
    final_capital  : float
    total_return   : float     # En %

    # Métriques de risque ajusté
    sharpe_ratio  : float
    sortino_ratio : float
    calmar_ratio  : float      # CAGR / |MaxDrawdown|

    # Risque
    max_drawdown  : float      # En % (négatif)
    volatility    : float      # Annualisée en %
    var_95        : float      # VaR 95% journalière

    # Métriques de trading
    total_trades  : int
    win_rate      : float      # En %
    profit_factor : float
    avg_win       : float      # Gain moyen en %
    avg_loss      : float      # Perte moyenne en %
    best_trade    : float      # Meilleur trade en %
    worst_trade   : float      # Pire trade en %

    # Performance
    cagr          : float      # Rendement annuel composé en %
    benchmark_return: float    # Rendement Buy & Hold en %

    # Série temporelle pour graphiques
    equity_curve  : pd.Series = field(default_factory=pd.Series)
    drawdown_series: pd.Series = field(default_factory=pd.Series)
    trades_df     : pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_dict(self) -> Dict:
        """Convertit en dictionnaire (sans les séries temporelles)."""
        return {
            "symbol"          : self.symbol,
            "période"         : f"{self.start_date} → {self.end_date}",
            "capital_initial" : f"{self.initial_capital:,.0f}$",
            "capital_final"   : f"{self.final_capital:,.0f}$",
            "rendement_total" : f"{self.total_return:+.1f}%",
            "sharpe_ratio"    : round(self.sharpe_ratio, 2),
            "sortino_ratio"   : round(self.sortino_ratio, 2),
            "max_drawdown"    : f"{self.max_drawdown:.1f}%",
            "volatilité"      : f"{self.volatility:.1f}%",
            "cagr"            : f"{self.cagr:.1f}%",
            "total_trades"    : self.total_trades,
            "win_rate"        : f"{self.win_rate:.1f}%",
            "profit_factor"   : round(self.profit_factor, 2),
            "benchmark"       : f"{self.benchmark_return:+.1f}%",
            "alpha"           : f"{self.total_return - self.benchmark_return:+.1f}%"
        }

    def print_report(self):
        """Affiche un rapport formaté dans les logs."""
        d = self.to_dict()
        logger.info("=" * 60)
        logger.info(f"RAPPORT DE BACKTESTING : {self.symbol}")
        logger.info("=" * 60)
        for key, val in d.items():
            logger.info(f"  {key:<22}: {val}")
        logger.info("=" * 60)

        # Évaluation
        passed = []
        failed = []
        cfg_check = [
            (self.sharpe_ratio >= 1.0,   f"Sharpe ≥ 1.0 ({self.sharpe_ratio:.2f})"),
            (self.max_drawdown >= -20.0, f"MaxDD ≥ -20% ({self.max_drawdown:.1f}%)"),
            (self.win_rate >= 50.0,      f"Win Rate ≥ 50% ({self.win_rate:.1f}%)"),
            (self.profit_factor >= 1.5,  f"Profit Factor ≥ 1.5 ({self.profit_factor:.2f})")
        ]
        for ok, msg in cfg_check:
            (passed if ok else failed).append(msg)

        logger.info(f"  CRITÈRES RÉUSSIS ({len(passed)}/{len(cfg_check)}) :")
        for m in passed:
            logger.info(f"    ✓ {m}")
        for m in failed:
            logger.info(f"    ✗ {m}")
        logger.info("=" * 60)


class Backtester:
    """
    Moteur de backtesting réaliste.

    UTILISATION :
        bt = Backtester(config)
        result = bt.run(df_ohlcv, signals, symbol="AAPL")
        result.print_report()

        # Walk-forward
        wf_results = bt.walk_forward(df_features, model, symbol="AAPL")
    """

    def __init__(self, config: dict):
        self.config = config
        self.bt_cfg = config["backtesting"]
        self.transaction_cost = self.bt_cfg["transaction_cost_pct"] / 100
        self.slippage         = self.bt_cfg["slippage_pct"] / 100
        self.initial_capital  = config["portfolio"]["initial_capital"]

    # ─────────────────────────────────────────────────────────────
    #  SIMULATION PRINCIPALE
    # ─────────────────────────────────────────────────────────────

    def run(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        symbol: str = "ASSET",
        position_size_pct: float = 0.10
    ) -> BacktestResult:
        """
        Simule la stratégie sur les données historiques.

        PARAMÈTRES :
        - df               : DataFrame OHLCV avec index datetime
        - signals          : Série de signaux (+1=achat, -1=vente, 0=hold)
        - symbol           : Nom de l'actif
        - position_size_pct: Taille de chaque position (% du capital)

        RETOURNE :
        - BacktestResult avec toutes les métriques calculées

        LOGIQUE DE SIMULATION :
        Pour chaque jour :
          1. Lire le signal du modèle (préparé le jour précédent)
          2. Exécuter l'ordre si nécessaire (avec frais + slippage)
          3. Marquer les stop-loss / take-profit
          4. Enregistrer la valeur du portefeuille
        """
        logger.info(f"Démarrage du backtesting : {symbol}")
        logger.info(f"Période : {df.index[0].date()} → {df.index[-1].date()}")

        # ── Initialisation ──
        capital         = float(self.initial_capital)
        shares          = 0.0
        entry_price     = 0.0
        entry_date      = None
        equity          = []
        trades          = []
        stop_loss_pct   = self.config["risk"]["stop_loss_pct"] / 100
        take_profit_pct = self.config["risk"]["take_profit_pct"] / 100

        prices = df["close"].values
        dates  = df.index

        # Aligner les signaux sur les dates du DataFrame
        signals_aligned = signals.reindex(df.index).fillna(0)

        for i, (date, price) in enumerate(zip(dates, prices)):
            # Calculer la valeur du portefeuille
            portfolio_val = capital + shares * price
            equity.append(portfolio_val)

            if i == 0:
                continue

            signal = float(signals_aligned.iloc[i])

            # ── Stop-Loss / Take-Profit ──
            if shares > 0 and entry_price > 0:
                pnl_pct = (price - entry_price) / entry_price

                if pnl_pct <= -stop_loss_pct:
                    # STOP-LOSS : vendre avec slippage défavorable
                    sell_price = price * (1 - self.slippage)
                    proceeds   = shares * sell_price * (1 - self.transaction_cost)
                    trades.append({
                        "entry_date"  : entry_date,
                        "exit_date"   : date,
                        "entry_price" : entry_price,
                        "exit_price"  : sell_price,
                        "pnl_pct"     : pnl_pct * 100,
                        "type"        : "STOP-LOSS"
                    })
                    capital     += proceeds
                    shares       = 0.0
                    entry_price  = 0.0
                    continue

                elif pnl_pct >= take_profit_pct:
                    # TAKE-PROFIT
                    sell_price = price * (1 - self.slippage)
                    proceeds   = shares * sell_price * (1 - self.transaction_cost)
                    trades.append({
                        "entry_date"  : entry_date,
                        "exit_date"   : date,
                        "entry_price" : entry_price,
                        "exit_price"  : sell_price,
                        "pnl_pct"     : pnl_pct * 100,
                        "type"        : "TAKE-PROFIT"
                    })
                    capital     += proceeds
                    shares       = 0.0
                    entry_price  = 0.0
                    continue

            # ── Exécution des signaux ──
            if signal > 0.5 and shares == 0:  # ACHETER
                buy_price    = price * (1 + self.slippage)
                invest       = capital * position_size_pct
                cost         = invest * (1 + self.transaction_cost)
                if cost <= capital:
                    shares_bought = invest / buy_price
                    shares       += shares_bought
                    capital      -= cost
                    entry_price   = buy_price
                    entry_date    = date

            elif signal < -0.5 and shares > 0:  # VENDRE
                sell_price = price * (1 - self.slippage)
                proceeds   = shares * sell_price * (1 - self.transaction_cost)
                pnl_pct    = (sell_price - entry_price) / entry_price if entry_price > 0 else 0
                trades.append({
                    "entry_date"  : entry_date,
                    "exit_date"   : date,
                    "entry_price" : entry_price,
                    "exit_price"  : sell_price,
                    "pnl_pct"     : pnl_pct * 100,
                    "type"        : "SIGNAL"
                })
                capital     += proceeds
                shares       = 0.0
                entry_price  = 0.0

        # Clôture finale si position ouverte
        if shares > 0:
            final_price = prices[-1] * (1 - self.slippage)
            proceeds    = shares * final_price * (1 - self.transaction_cost)
            pnl_pct     = (final_price - entry_price) / entry_price if entry_price > 0 else 0
            trades.append({
                "entry_date"  : entry_date,
                "exit_date"   : dates[-1],
                "entry_price" : entry_price,
                "exit_price"  : final_price,
                "pnl_pct"     : pnl_pct * 100,
                "type"        : "CLOSE"
            })
            capital += proceeds

        # ── Calcul des métriques ──
        equity_series = pd.Series(equity, index=dates)
        return self._compute_metrics(
            equity_series, trades, symbol,
            benchmark_start=prices[0], benchmark_end=prices[-1]
        )

    # ─────────────────────────────────────────────────────────────
    #  CALCUL DES MÉTRIQUES
    # ─────────────────────────────────────────────────────────────

    def _compute_metrics(
        self,
        equity: pd.Series,
        trades: List[Dict],
        symbol: str,
        benchmark_start: float,
        benchmark_end: float
    ) -> BacktestResult:
        """Calcule toutes les métriques à partir de la courbe d'équité."""

        final_capital  = float(equity.iloc[-1])
        total_return   = (final_capital - self.initial_capital) / self.initial_capital * 100
        n_days         = len(equity)
        n_years        = n_days / 252.0

        # CAGR : Rendement Annuel Composé
        cagr = ((final_capital / self.initial_capital) ** (1 / max(n_years, 0.01)) - 1) * 100

        # Rendements journaliers
        daily_rets = equity.pct_change().dropna()
        vol = float(daily_rets.std() * np.sqrt(252) * 100)

        # Sharpe Ratio — retourner 0 si pas de variance (aucun trade / pas de mouvement)
        risk_free_daily = 0.05 / 252
        excess_rets = daily_rets - risk_free_daily
        ret_std = float(excess_rets.std())
        if ret_std < 1e-6:
            sharpe = 0.0
        else:
            sharpe = float(excess_rets.mean() / ret_std * np.sqrt(252))
        sharpe = float(np.clip(sharpe, -50, 50))  # Borner pour lisibilité

        # Sortino Ratio (pénalise seulement les rendements négatifs)
        downside_rets = daily_rets[daily_rets < 0]
        downside_std  = float(downside_rets.std() * np.sqrt(252)) if len(downside_rets) > 0 else 1e-8
        sortino_raw = float(excess_rets.mean() * 252 / (downside_std + 1e-8))
        sortino = float(np.clip(sortino_raw, -50, 50))

        # Drawdown
        cummax  = equity.cummax()
        dd_series = (equity - cummax) / (cummax + 1e-8) * 100
        max_dd  = float(dd_series.min())

        # Calmar Ratio
        calmar = cagr / (abs(max_dd) + 1e-8)

        # VaR 95%
        var_95 = float(abs(np.percentile(daily_rets * 100, 5)))

        # Statistiques des trades
        trades_df    = pd.DataFrame(trades) if trades else pd.DataFrame()
        total_trades = len(trades)
        win_rate     = 0.0
        profit_factor = 0.0
        avg_win = avg_loss = best_trade = worst_trade = 0.0

        if total_trades > 0 and "pnl_pct" in trades_df.columns:
            pnls     = trades_df["pnl_pct"].values
            wins     = pnls[pnls > 0]
            losses   = pnls[pnls <= 0]
            win_rate = len(wins) / total_trades * 100
            avg_win  = float(wins.mean()) if len(wins) > 0 else 0.0
            avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
            total_gain = wins.sum() if len(wins) > 0 else 0.0
            total_loss = abs(losses.sum()) if len(losses) > 0 else 1e-8
            profit_factor = total_gain / total_loss
            best_trade  = float(pnls.max()) if len(pnls) > 0 else 0.0
            worst_trade = float(pnls.min()) if len(pnls) > 0 else 0.0

        # Benchmark Buy & Hold
        benchmark_return = (benchmark_end - benchmark_start) / (benchmark_start + 1e-8) * 100

        result = BacktestResult(
            symbol=symbol,
            start_date=str(equity.index[0].date()),
            end_date=str(equity.index[-1].date()),
            n_days=n_days,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            volatility=vol,
            var_95=var_95,
            total_trades=total_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            best_trade=best_trade,
            worst_trade=worst_trade,
            cagr=cagr,
            benchmark_return=benchmark_return,
            equity_curve=equity,
            drawdown_series=dd_series,
            trades_df=trades_df
        )

        result.print_report()
        return result

    # ─────────────────────────────────────────────────────────────
    #  WALK-FORWARD VALIDATION
    # ─────────────────────────────────────────────────────────────

    def walk_forward(
        self,
        df: pd.DataFrame,
        signal_func,
        symbol: str = "ASSET"
    ) -> List[BacktestResult]:
        """
        Validation walk-forward : la méthode la plus robuste de backtesting.

        COMMENT ÇA MARCHE :
        On divise les données en fenêtres glissantes.
        Chaque fenêtre a une période d'entraînement ET une période de test.

        EXEMPLE avec config par défaut :
        Fenêtre 1: Train [252 jours] → Test [63 jours]
        Fenêtre 2: Train [252 jours] → Test [63 jours]  (décalé de 21 jours)
        Fenêtre 3: Train [252 jours] → Test [63 jours]  (décalé de 21 jours)
        ...

        POURQUOI C'EST PLUS ROBUSTE QU'UN BACKTESTING SIMPLE ?
        On évite d'entraîner et tester sur la même période.
        La validation croisée temporelle donne une estimation
        plus fiable des performances futures réelles.

        PARAMÈTRES :
        - df          : DataFrame OHLCV complet
        - signal_func : Fonction f(train_df, test_df) → pd.Series de signaux
        - symbol      : Nom de l'actif

        RETOURNE :
        - Liste de BacktestResult (une par fenêtre)
        """
        wf_cfg     = self.bt_cfg["walk_forward"]
        if not wf_cfg.get("enable", True):
            logger.info("Walk-forward désactivé → backtesting simple")
            return []

        train_w = wf_cfg["train_window_days"]
        test_w  = wf_cfg["test_window_days"]
        step    = wf_cfg["step_days"]

        results = []
        n = len(df)
        starts = range(0, n - train_w - test_w, step)

        logger.info(f"Walk-Forward Validation : {len(list(starts))} fenêtres")

        for i, start in enumerate(starts):
            train_end = start + train_w
            test_end  = train_end + test_w

            if test_end > n:
                break

            train_df = df.iloc[start:train_end]
            test_df  = df.iloc[train_end:test_end]

            logger.info(
                f"  Fenêtre {i+1} : "
                f"Train [{train_df.index[0].date()} → {train_df.index[-1].date()}] | "
                f"Test  [{test_df.index[0].date()} → {test_df.index[-1].date()}]"
            )

            try:
                # Générer les signaux avec la fonction fournie
                signals = signal_func(train_df, test_df)

                # Backtest sur la fenêtre de test
                result = self.run(test_df, signals, symbol=f"{symbol}_wf{i+1}")
                results.append(result)

            except Exception as e:
                logger.error(f"  Fenêtre {i+1} : Erreur → {e}")

        # Résumé agrégé
        if results:
            self._summarize_walk_forward(results, symbol)

        return results

    def _summarize_walk_forward(
        self,
        results: List[BacktestResult],
        symbol: str
    ):
        """Affiche un résumé agrégé des résultats walk-forward."""
        sharpes  = [r.sharpe_ratio for r in results]
        returns  = [r.total_return for r in results]
        max_dds  = [r.max_drawdown for r in results]
        win_rates = [r.win_rate for r in results]

        logger.info("=" * 60)
        logger.info(f"RÉSUMÉ WALK-FORWARD : {symbol} ({len(results)} fenêtres)")
        logger.info("=" * 60)
        logger.info(f"  Sharpe moyen     : {np.mean(sharpes):+.2f} (±{np.std(sharpes):.2f})")
        logger.info(f"  Rendement moyen  : {np.mean(returns):+.1f}%")
        logger.info(f"  MaxDD moyen      : {np.mean(max_dds):.1f}%")
        logger.info(f"  Win Rate moyen   : {np.mean(win_rates):.1f}%")
        logger.info(f"  Fenêtres + / -   : {sum(r > 0 for r in returns)} / {sum(r <= 0 for r in returns)}")
        logger.info("=" * 60)

    def save_results(self, result: BacktestResult, path: str = "./results/backtest.json"):
        """Sauvegarde les résultats en JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"Résultats sauvegardés : {path}")
