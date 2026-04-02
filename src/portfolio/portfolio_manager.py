"""
GESTIONNAIRE DE PORTEFEUILLE  (Couche 7)
==========================================

QU'EST-CE QUE LA GESTION DE PORTEFEUILLE ?
C'est l'art de décider COMMENT répartir le capital entre
plusieurs actifs pour maximiser le rendement et minimiser le risque.

THÉORIE MODERNE DU PORTEFEUILLE (Markowitz, 1952) :
"Ne mettez pas tous vos œufs dans le même panier."
En combinant des actifs peu corrélés, on peut obtenir
le même rendement avec MOINS de risque.

ANALOGIE :
- Portefeuille 100% AAPL : Si Apple chute de 30%, vous perdez 30%
- Portefeuille AAPL + BTC + Or : Ils ne chutent pas tous en même temps
  → Le portefeuille diversifié est plus stable

ALGORITHMES D'OPTIMISATION :
1. Maximum Sharpe Ratio : Maximiser rendement/risque
2. Minimum Variance     : Minimiser le risque total
3. Equal Weight         : 1/N dans chaque actif (simple et robuste)
4. Risk Parity          : Chaque actif contribue également au risque

RÉÉQUILIBRAGE DYNAMIQUE :
Les poids optimaux changent avec le temps. On rééquilibre
périodiquement (chaque semaine par défaut) pour maintenir
l'allocation cible.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.optimize import minimize
from pathlib import Path

from ..utils.logger import get_logger

logger = get_logger("PortfolioManager")


class PortfolioManager:
    """
    Optimise et gère l'allocation du capital entre les actifs.

    UTILISATION :
        pm = PortfolioManager(config)

        # Calculer les poids optimaux
        weights = pm.optimize(returns_df, method="max_sharpe")

        # Calculer les ordres de rééquilibrage
        orders = pm.rebalance(current_positions, target_weights, portfolio_value)
    """

    def __init__(self, config: dict):
        self.config    = config
        self.cfg       = config["portfolio"]
        self.initial_capital = self.cfg["initial_capital"]
        self.cash      = float(self.initial_capital)
        self.positions : Dict[str, float] = {}    # {symbol: valeur en $}
        self.weights   : Dict[str, float] = {}    # {symbol: poids cible}
        self.trade_log : List[Dict] = []

    # ─────────────────────────────────────────────────────────────
    #  CALCUL DES STATISTIQUES DE PORTEFEUILLE
    # ─────────────────────────────────────────────────────────────

    def compute_expected_returns(self, prices_df: pd.DataFrame) -> pd.Series:
        """
        Calcule les rendements espérés annualisés pour chaque actif.

        MÉTHODE :
        On utilise la moyenne des rendements historiques comme estimateur
        des rendements futurs (hypothèse simplificatrice).

        NOTE : En pratique, les hedge funds utilisent des modèles
        factoriels (CAPM, Fama-French) pour de meilleures estimations.

        PARAMÈTRE :
        - prices_df : DataFrame avec une colonne par actif, index = dates

        RETOURNE :
        - Série des rendements annualisés par actif
        """
        daily_returns = prices_df.pct_change().dropna()
        annual_returns = daily_returns.mean() * 252
        return annual_returns

    def compute_covariance_matrix(
        self,
        prices_df: pd.DataFrame,
        method: str = "ledoit_wolf"
    ) -> pd.DataFrame:
        """
        Calcule la matrice de covariance des rendements.

        QU'EST-CE QUE LA MATRICE DE COVARIANCE ?
        C'est une matrice qui résume comment les actifs bougent
        ensemble (corrélation) et leur volatilité individuelle.

        EXEMPLE (2 actifs : AAPL et BTC) :
        ┌────────────┬──────┬──────┐
        │            │ AAPL │ BTC  │
        ├────────────┼──────┼──────┤
        │ AAPL       │ 0.04 │ 0.01 │  ← Variance AAPL (diagonale)
        │ BTC        │ 0.01 │ 0.25 │  ← Covariance AAPL/BTC (hors diagonale)
        └────────────┴──────┴──────┘
        → BTC est plus volatile (0.25 > 0.04)
        → Faible corrélation AAPL/BTC (0.01 / √(0.04×0.25) = 0.10)

        MÉTHODE LEDOIT-WOLF :
        Une amélioration de la covariance empirique classique.
        Plus robuste avec peu de données (shrinkage estimator).

        PARAMÈTRES :
        - prices_df : DataFrame des prix
        - method    : "ledoit_wolf" (recommandé) ou "empirical"
        """
        daily_returns = prices_df.pct_change().dropna()

        if method == "ledoit_wolf":
            try:
                from sklearn.covariance import LedoitWolf
                lw = LedoitWolf()
                lw.fit(daily_returns.dropna())
                cov_matrix = pd.DataFrame(
                    lw.covariance_ * 252,  # Annualiser
                    index=daily_returns.columns,
                    columns=daily_returns.columns
                )
                return cov_matrix
            except Exception:
                pass  # Fallback sur empirique

        # Covariance empirique annualisée
        return daily_returns.cov() * 252

    # ─────────────────────────────────────────────────────────────
    #  OPTIMISATION DES POIDS
    # ─────────────────────────────────────────────────────────────

    def _portfolio_sharpe(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_free_rate: float = 0.05
    ) -> float:
        """
        Calcule le ratio de Sharpe d'un portefeuille avec des poids donnés.

        FORMULE :
        Sharpe = (Rendement_portefeuille - Taux_sans_risque) / Volatilité_portefeuille

        Rendement_portefeuille  = w^T × μ
        Variance_portefeuille   = w^T × Σ × w
        Volatilité_portefeuille = √(w^T × Σ × w)

        ON RETOURNE LE NÉGATIF car minimize() minimise (on veut maximiser).
        """
        port_return = float(np.dot(weights, expected_returns))
        port_var    = float(np.dot(weights.T, np.dot(cov_matrix, weights)))
        port_vol    = np.sqrt(max(port_var, 1e-10))
        sharpe      = (port_return - risk_free_rate) / port_vol
        return -sharpe  # Négatif pour minimisation

    def _portfolio_variance(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray
    ) -> float:
        """Calcule la variance du portefeuille (pour min-variance)."""
        return float(np.dot(weights.T, np.dot(cov_matrix, weights)))

    def optimize(
        self,
        prices_df: pd.DataFrame,
        method: Optional[str] = None,
        regime: str = "BULL"
    ) -> Dict[str, float]:
        """
        Calcule les poids optimaux du portefeuille.

        PARAMÈTRES :
        - prices_df : DataFrame des prix de clôture (colonne = actif)
        - method    : "max_sharpe" / "min_volatility" / "equal_weight" / "risk_parity"
        - regime    : Régime de marché (peut influencer la méthode)

        RETOURNE :
        - Dict {symbole: poids} où Σ poids = 1.0

        EXEMPLE :
        {"AAPL": 0.25, "BTC-USD": 0.15, "GC=F": 0.20, "MSFT": 0.20, ...}
        """
        method = method or self.cfg["optimization_method"]
        symbols = prices_df.columns.tolist()
        n = len(symbols)

        if n == 0:
            return {}

        # En régime HIGH_VOL → forcer equal weight (plus prudent)
        if regime == "HIGH_VOL":
            method = "equal_weight"
            logger.info("Régime HIGH_VOL → allocation équipondérée forcée")

        # ── Poids égaux (fallback et référence) ──
        if method == "equal_weight" or n < 3:
            w = 1.0 / n
            weights = {sym: w for sym in symbols}
            logger.info(f"Allocation équipondérée : {w:.1%} × {n} actifs")
            self.weights = weights
            return weights

        # ── Calcul des inputs d'optimisation ──
        mu  = self.compute_expected_returns(prices_df).values
        cov = self.compute_covariance_matrix(prices_df).values

        # Contraintes :
        # 1. Somme des poids = 1
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        # 2. Poids entre min_weight et max_weight
        bounds = tuple(
            (self.cfg["min_weight"], self.cfg["max_weight"])
            for _ in range(n)
        )

        # Point de départ : poids égaux
        w0 = np.array([1.0 / n] * n)

        try:
            if method == "max_sharpe":
                result = minimize(
                    self._portfolio_sharpe,
                    w0, args=(mu, cov),
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints,
                    options={"maxiter": 1000, "ftol": 1e-9}
                )
                opt_weights = result.x

            elif method == "min_volatility":
                result = minimize(
                    self._portfolio_variance,
                    w0, args=(cov,),
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints,
                    options={"maxiter": 1000, "ftol": 1e-9}
                )
                opt_weights = result.x

            elif method == "risk_parity":
                opt_weights = self._risk_parity(cov, n, bounds, constraints)

            else:
                logger.warning(f"Méthode '{method}' inconnue → equal weight")
                opt_weights = w0

            # Normaliser (s'assurer que la somme = 1)
            opt_weights = np.clip(opt_weights, 0, None)
            opt_weights /= opt_weights.sum()

            weights = {sym: float(w) for sym, w in zip(symbols, opt_weights)}

            # Rapport
            port_ret = float(np.dot(opt_weights, mu))
            port_vol = float(np.sqrt(np.dot(opt_weights.T, np.dot(cov, opt_weights))))
            port_sharpe = (port_ret - 0.05) / (port_vol + 1e-8)

            logger.info(f"Optimisation ({method}) :")
            logger.info(f"  Rendement attendu : {port_ret:.1%}")
            logger.info(f"  Volatilité        : {port_vol:.1%}")
            logger.info(f"  Sharpe ratio      : {port_sharpe:.2f}")

        except Exception as e:
            logger.error(f"Optimisation échouée ({e}) → equal weight")
            weights = {sym: 1.0 / n for sym in symbols}

        self.weights = weights
        return weights

    def _risk_parity(
        self,
        cov: np.ndarray,
        n: int,
        bounds: tuple,
        constraints: list
    ) -> np.ndarray:
        """
        Optimisation Risk Parity : chaque actif contribue également au risque.

        CONCEPT :
        Au lieu de maximiser le Sharpe, on cherche des poids tels que
        chaque actif contribue exactement 1/N du risque total.

        Cela donne naturellement plus de poids aux actifs peu volatils
        (comme les obligations, l'or) et moins aux actifs volatils (crypto).

        RÉSULTAT TYPIQUE :
        Un portefeuille Risk Parity alloue ~40% obligations, ~30% actions,
        ~20% or, ~10% matières premières.
        C'est la base du "All-Weather Portfolio" de Ray Dalio.
        """
        def risk_parity_objective(w):
            """Minimise la variance des contributions au risque."""
            w = np.abs(w)
            port_var = np.dot(w.T, np.dot(cov, w))
            # Contribution marginale au risque de chaque actif
            marginal_contrib = np.dot(cov, w)
            risk_contrib = w * marginal_contrib / (port_var + 1e-10)
            target = 1.0 / n
            return np.sum((risk_contrib - target) ** 2)

        result = minimize(
            risk_parity_objective, np.ones(n) / n,
            method="SLSQP", bounds=bounds, constraints=constraints,
            options={"maxiter": 2000, "ftol": 1e-12}
        )
        return result.x

    # ─────────────────────────────────────────────────────────────
    #  RÉÉQUILIBRAGE
    # ─────────────────────────────────────────────────────────────

    def rebalance(
        self,
        current_positions: Dict[str, float],
        target_weights: Dict[str, float],
        portfolio_value: float,
        transaction_cost: float = 0.001
    ) -> Dict[str, float]:
        """
        Calcule les ordres nécessaires pour atteindre les poids cibles.

        POURQUOI RÉÉQUILIBRER ?
        Avec le temps, les actifs performants prennent plus de place.
        Ex: AAPL monte de 30% → sa part passe de 20% à 24%.
        On rééquilibre pour revenir aux poids cibles.

        EXEMPLE :
        Capital   = 100 000$
        Cible AAPL = 25%  → 25 000$
        Actuel    = 28 000$ (a monté)
        Ordre     : VENDRE 3 000$ d'AAPL

        PARAMÈTRES :
        - current_positions : {symbol: valeur actuelle en $}
        - target_weights    : {symbol: poids cible 0.0-1.0}
        - portfolio_value   : Valeur totale du portefeuille ($)
        - transaction_cost  : Coût de transaction (ex: 0.001 = 0.1%)

        RETOURNE :
        - Dict {symbol: montant en $}
          Positif = ACHETER | Négatif = VENDRE
        """
        orders = {}
        cash_buffer = self.cfg.get("cash_buffer_pct", 5.0) / 100.0
        investable = portfolio_value * (1 - cash_buffer)

        for symbol, target_w in target_weights.items():
            target_value  = investable * target_w
            current_value = current_positions.get(symbol, 0.0)
            delta = target_value - current_value

            # Ne rééquilibrer que si l'écart est significatif (> 1%)
            # → Éviter des coûts de transaction pour de micro-ajustements
            if abs(delta) / (portfolio_value + 1e-8) > 0.01:
                # Coût de transaction
                cost = abs(delta) * transaction_cost
                if delta > 0:
                    orders[symbol] = delta - cost  # Achat net
                else:
                    orders[symbol] = delta + cost  # Vente nette

        if orders:
            logger.info(f"Rééquilibrage : {len(orders)} ordres générés")
            for sym, amount in orders.items():
                action = "ACHAT" if amount > 0 else "VENTE"
                logger.info(f"  {action} {sym}: {abs(amount):,.0f}$")

        return orders

    # ─────────────────────────────────────────────────────────────
    #  RAPPORT DU PORTEFEUILLE
    # ─────────────────────────────────────────────────────────────

    def get_portfolio_summary(
        self,
        current_prices: Dict[str, float],
        shares: Dict[str, float]
    ) -> Dict:
        """
        Génère un résumé complet du portefeuille.

        RETOURNE :
        {
            "total_value"    : Valeur totale en $
            "cash"           : Cash disponible en $
            "invested"       : Montant investi en $
            "total_return"   : Rendement total depuis le début
            "positions"      : Liste des positions détaillées
        }
        """
        invested_value = sum(
            shares.get(sym, 0) * price
            for sym, price in current_prices.items()
        )
        total_value = self.cash + invested_value
        total_return = (total_value - self.initial_capital) / self.initial_capital * 100

        positions = []
        for sym, price in current_prices.items():
            qty = shares.get(sym, 0)
            if qty > 0:
                value = qty * price
                weight = value / (total_value + 1e-8) * 100
                positions.append({
                    "symbol": sym,
                    "shares": round(qty, 4),
                    "price" : round(price, 2),
                    "value" : round(value, 2),
                    "weight": round(weight, 1)
                })

        return {
            "total_value"  : round(total_value, 2),
            "cash"         : round(self.cash, 2),
            "invested"     : round(invested_value, 2),
            "total_return" : round(total_return, 2),
            "n_positions"  : len(positions),
            "positions"    : sorted(positions, key=lambda x: -x["value"])
        }
