"""
GESTIONNAIRE DE RISQUES  (Couche 6 — CRITIQUE)
================================================

AVERTISSEMENT IMPORTANT :
La gestion des risques est la composante LA PLUS CRITIQUE
d'un système de trading. Un système sans gestion des risques
peut perdre tout le capital en quelques transactions.

PHILOSOPHIE :
"La première règle du trading est de ne pas perdre d'argent.
La deuxième règle est de ne pas oublier la première règle."
— Warren Buffett (adapté)

NIVEAUX DE PROTECTION IMPLÉMENTÉS :

Niveau 1 — Limites par POSITION :
  - Taille maximale par actif (10% du capital)
  - Stop-loss automatique (-5%)
  - Take-profit automatique (+15%)

Niveau 2 — Limites du PORTEFEUILLE :
  - Drawdown maximum (-15%)
  - Perte journalière maximale (-3%)
  - VaR (Value at Risk) journalière (2%)

Niveau 3 — CIRCUIT BREAKERS (arrêt d'urgence) :
  - Krach de marché (-5% en 1 jour sur l'indice)
  - 5 pertes consécutives → pause forcée
  - Régime HIGH_VOL → réduction automatique

Niveau 4 — CORRÉLATION :
  - Éviter les positions trop corrélées
  - Limiter l'exposition sectorielle (30%)

STATUS DU SYSTÈME :
  GREEN  → Trading normal
  YELLOW → Alertes, positions réduites
  RED    → Trading suspendu (circuit breaker)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from ..utils.logger import get_logger

logger = get_logger("RiskManager")


class RiskStatus(Enum):
    """
    Statut global du système de trading.

    GREEN  = Tout va bien, trading normal
    YELLOW = Alertes détectées, positions réduites de 50%
    RED    = Circuit breaker déclenché, trading SUSPENDU
    """
    GREEN  = "GREEN"
    YELLOW = "YELLOW"
    RED    = "RED"


@dataclass
class PositionRisk:
    """
    Évaluation du risque pour une position individuelle.

    CHAMPS :
    - symbol         : Nom de l'actif (ex: "AAPL")
    - size_pct       : Taille en % du portefeuille
    - unrealized_pnl : Profit/perte non réalisé en %
    - stop_loss_pct  : Niveau de stop-loss en %
    - risk_level     : LOW / MEDIUM / HIGH
    - alerts         : Liste des alertes actives
    """
    symbol        : str
    size_pct      : float
    unrealized_pnl: float
    stop_loss_pct : float
    risk_level    : str
    alerts        : List[str] = field(default_factory=list)


@dataclass
class PortfolioRisk:
    """
    Évaluation du risque global du portefeuille.
    """
    total_exposure_pct : float      # % du capital investi
    current_drawdown   : float      # Drawdown actuel en %
    daily_var          : float      # VaR journalière en %
    correlation_risk   : float      # Risque de corrélation (0-1)
    status             : RiskStatus
    alerts             : List[str]
    position_risks     : List[PositionRisk]
    can_trade          : bool       # L'agent peut-il ouvrir de nouvelles positions ?
    position_multiplier: float      # Multiplicateur de taille (0.0 → 1.0)


class RiskManager:
    """
    Gestionnaire de risques institutionnel.

    UTILISATION :
        rm = RiskManager(config)

        # Avant chaque trade
        allowed, size, reason = rm.check_trade(symbol, signal, portfolio)

        # Après chaque période
        risk_report = rm.evaluate_portfolio(portfolio, market_data)
    """

    def __init__(self, config: dict):
        self.config = config
        self.cfg    = config["risk"]
        self.consecutive_losses = 0
        self.daily_trades: List[Dict] = []
        self.status = RiskStatus.GREEN
        self._log_limits()

    def _log_limits(self):
        """Affiche les limites de risque configurées au démarrage."""
        logger.info("=" * 55)
        logger.info("LIMITES DE RISQUE ACTIVES :")
        logger.info(f"  Drawdown max        : -{self.cfg['max_drawdown_pct']:.0f}%")
        logger.info(f"  Perte journalière   : -{self.cfg['max_daily_loss_pct']:.0f}%")
        logger.info(f"  Taille max/position : {self.cfg['max_position_size_pct']:.0f}%")
        logger.info(f"  Stop-loss           : -{self.cfg['stop_loss_pct']:.0f}%")
        logger.info(f"  Take-profit         : +{self.cfg['take_profit_pct']:.0f}%")
        logger.info("=" * 55)

    # ─────────────────────────────────────────────────────────────
    #  VÉRIFICATION AVANT TRADE
    # ─────────────────────────────────────────────────────────────

    def check_trade(
        self,
        symbol: str,
        signal: str,
        portfolio_value: float,
        current_positions: Dict[str, float],
        volatility: float = 0.15,
        regime: str = "BULL"
    ) -> Tuple[bool, float, str]:
        """
        Vérifie si un trade est autorisé et calcule la taille optimale.

        PARAMÈTRES :
        - symbol             : Actif à trader (ex: "AAPL")
        - signal             : "BUY", "SELL", ou "HOLD"
        - portfolio_value    : Valeur totale du portefeuille ($)
        - current_positions  : Dict {symbole: valeur en $}
        - volatility         : Volatilité récente de l'actif (annualisée)
        - regime             : Régime de marché actuel

        RETOURNE :
        - allowed   : True si le trade est autorisé
        - size_pct  : Taille recommandée en % du portefeuille (0.0 → 1.0)
        - reason    : Explication de la décision

        EXEMPLE :
        allowed, size, reason = rm.check_trade("AAPL", "BUY", 100000, {}, 0.20, "BULL")
        → (True, 0.08, "Trade autorisé. Taille: 8.0% (réduite pour volatilité)")
        """
        # ── Circuit breaker global ──
        if self.status == RiskStatus.RED:
            return False, 0.0, "CIRCUIT BREAKER ACTIF : Trading suspendu"

        if signal == "HOLD":
            return False, 0.0, "Signal HOLD : aucune action requise"

        # ── Taille de base ──
        base_size = self.cfg["max_position_size_pct"] / 100.0

        # ── Ajustements selon le régime ──
        regime_multipliers = {
            "BULL"    : 1.0,
            "BEAR"    : 0.5,   # Réduire de 50% en marché baissier
            "RANGE"   : 0.75,
            "HIGH_VOL": 0.25,  # Réduire de 75% en haute volatilité
        }
        regime_mult = regime_multipliers.get(regime, 0.75)

        # ── Ajustement selon la volatilité (Position Sizing de Kelly) ──
        # Plus la volatilité est élevée, plus la position est petite
        target_vol = 0.15  # Volatilité cible annualisée = 15%
        vol_mult = min(1.0, target_vol / (volatility + 1e-8))
        vol_mult = np.clip(vol_mult, 0.2, 1.0)

        # ── Vérifier la limite par position ──
        existing = current_positions.get(symbol, 0.0)
        existing_pct = existing / (portfolio_value + 1e-8)
        if existing_pct >= self.cfg["max_position_size_pct"] / 100.0:
            return False, 0.0, f"{symbol} : Limite de position atteinte ({existing_pct:.1%})"

        # ── Vérifier la concentration totale ──
        total_invested = sum(current_positions.values())
        total_pct = total_invested / (portfolio_value + 1e-8)
        if total_pct > 0.90:  # Max 90% investi
            return False, 0.0, f"Portfolio saturé ({total_pct:.1%} investi)"

        # ── Réduction si YELLOW ──
        status_mult = 0.5 if self.status == RiskStatus.YELLOW else 1.0

        # ── Taille finale ──
        final_size = base_size * regime_mult * vol_mult * status_mult
        final_size = np.clip(final_size, 0.01, self.cfg["max_position_size_pct"] / 100.0)

        reason = (
            f"Trade autorisé | Taille: {final_size:.1%} "
            f"(régime={regime_mult:.0%}, vol={vol_mult:.0%}, statut={status_mult:.0%})"
        )

        logger.info(f"[RiskManager] {symbol} {signal} → {reason}")
        return True, final_size, reason

    # ─────────────────────────────────────────────────────────────
    #  CALCUL DU DRAWDOWN
    # ─────────────────────────────────────────────────────────────

    def compute_drawdown(
        self,
        portfolio_values: pd.Series
    ) -> Tuple[float, float]:
        """
        Calcule le drawdown courant et le drawdown maximum.

        QU'EST-CE QUE LE DRAWDOWN ?
        Chute depuis le dernier sommet du portefeuille.

        EXEMPLE :
        Portefeuille : 100K → 120K → 95K → 110K
        Sommet       : 120K
        Drawdown     : (95K - 120K) / 120K = -20.8%

        RETOURNE :
        - current_dd : Drawdown actuel en % (négatif)
        - max_dd     : Drawdown maximum historique en % (négatif)
        """
        if len(portfolio_values) == 0:
            return 0.0, 0.0

        cummax = portfolio_values.cummax()
        drawdowns = (portfolio_values - cummax) / (cummax + 1e-8)

        current_dd = float(drawdowns.iloc[-1]) * 100
        max_dd     = float(drawdowns.min()) * 100

        return current_dd, max_dd

    # ─────────────────────────────────────────────────────────────
    #  CALCUL DE LA VaR
    # ─────────────────────────────────────────────────────────────

    def compute_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
        horizon_days: int = 1
    ) -> float:
        """
        Calcule la Value at Risk (VaR) historique.

        QU'EST-CE QUE LA VaR ?
        "Avec 95% de confiance, je ne perdrai pas plus de X%
        sur les 24 prochaines heures."

        MÉTHODE HISTORIQUE :
        On trie les rendements passés et on prend le percentile
        correspondant au niveau de confiance.

        EXEMPLE :
        500 jours de rendements → trier du pire au meilleur
        VaR 95% = rendement au rang 25 (5% des pires jours)
        Si ce rendement = -2.5% → VaR = 2.5%

        PARAMÈTRES :
        - returns    : Série des rendements historiques
        - confidence : Niveau de confiance (0.95 = 95%)
        - horizon    : Horizon en jours (×√horizon pour horizon > 1)

        RETOURNE :
        - VaR en % (positif, représente une perte potentielle)
        """
        if len(returns) < 20:
            return 0.0

        # VaR historique : percentile empirique
        var_1d = float(np.percentile(returns.dropna(), (1 - confidence) * 100))

        # Mise à l'échelle pour l'horizon donné (racine carrée du temps)
        var_scaled = var_1d * np.sqrt(horizon_days)

        return abs(var_scaled) * 100  # En %

    # ─────────────────────────────────────────────────────────────
    #  STOP-LOSS ET TAKE-PROFIT
    # ─────────────────────────────────────────────────────────────

    def check_stop_loss(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        position_size: float
    ) -> Tuple[bool, str]:
        """
        Vérifie si le stop-loss ou take-profit est atteint.

        STOP-LOSS :
        Ordre automatique qui clôture la position si le prix
        chute en dessous d'un seuil → limiter les pertes.

        TAKE-PROFIT :
        Ordre automatique qui clôture la position si le prix
        monte au-dessus d'un seuil → sécuriser les gains.

        PARAMÈTRES :
        - symbol        : Nom de l'actif
        - entry_price   : Prix d'achat initial ($)
        - current_price : Prix actuel ($)
        - position_size : Taille de la position en $ (pour le PnL)

        RETOURNE :
        - should_close : True si la position doit être clôturée
        - reason       : Raison de la clôture
        """
        if entry_price <= 0:
            return False, ""

        pnl_pct = (current_price - entry_price) / entry_price * 100

        # Stop-loss
        if pnl_pct <= -self.cfg["stop_loss_pct"]:
            self.consecutive_losses += 1
            logger.warning(
                f"STOP-LOSS déclenché | {symbol} | "
                f"PnL={pnl_pct:.1f}% | Pertes consécutives={self.consecutive_losses}"
            )
            return True, f"STOP-LOSS: {pnl_pct:.1f}% (seuil: -{self.cfg['stop_loss_pct']}%)"

        # Take-profit
        if pnl_pct >= self.cfg["take_profit_pct"]:
            self.consecutive_losses = 0  # Reset le compteur
            logger.info(f"TAKE-PROFIT atteint | {symbol} | PnL={pnl_pct:.1f}%")
            return True, f"TAKE-PROFIT: +{pnl_pct:.1f}% (seuil: +{self.cfg['take_profit_pct']}%)"

        return False, ""

    # ─────────────────────────────────────────────────────────────
    #  CIRCUIT BREAKERS
    # ─────────────────────────────────────────────────────────────

    def check_circuit_breakers(
        self,
        current_drawdown_pct: float,
        daily_loss_pct: float,
        market_daily_return: float = 0.0
    ) -> Tuple[RiskStatus, List[str]]:
        """
        Évalue tous les circuit breakers et met à jour le statut.

        CIRCUIT BREAKERS :
        Ce sont des disjoncteurs automatiques qui stoppent le trading
        quand le risque devient inacceptable.

        ┌─────────────────────────────────────────┬────────┐
        │ Condition                               │ Action │
        ├─────────────────────────────────────────┼────────┤
        │ Drawdown > max_drawdown_pct             │  RED   │
        │ Perte journalière > max_daily_loss_pct  │  RED   │
        │ Krach de marché (SPY < -5%)             │  RED   │
        │ 5+ pertes consécutives                  │  RED   │
        │ Drawdown > 60% du max                   │ YELLOW │
        │ Perte journalière > 60% du max          │ YELLOW │
        └─────────────────────────────────────────┴────────┘

        PARAMÈTRES :
        - current_drawdown_pct  : Drawdown actuel (négatif, ex: -8.5)
        - daily_loss_pct        : Perte du jour (négatif, ex: -2.1)
        - market_daily_return   : Rendement journalier de l'indice de référence

        RETOURNE :
        - Nouveau statut (GREEN / YELLOW / RED)
        - Liste des alertes actives
        """
        alerts = []
        new_status = RiskStatus.GREEN
        cb_cfg = self.cfg.get("circuit_breakers", {})

        # ── Vérifications RED ──
        if current_drawdown_pct <= -self.cfg["max_drawdown_pct"]:
            alerts.append(
                f"DRAWDOWN MAX ATTEINT : {current_drawdown_pct:.1f}% "
                f"(limite: -{self.cfg['max_drawdown_pct']}%)"
            )
            new_status = RiskStatus.RED

        if daily_loss_pct <= -self.cfg["max_daily_loss_pct"]:
            alerts.append(
                f"PERTE JOURNALIÈRE MAX : {daily_loss_pct:.1f}% "
                f"(limite: -{self.cfg['max_daily_loss_pct']}%)"
            )
            new_status = RiskStatus.RED

        if (cb_cfg.get("enable", True) and
                market_daily_return <= cb_cfg.get("market_crash_threshold", -5.0) / 100):
            alerts.append(
                f"KRACH DE MARCHÉ DÉTECTÉ : {market_daily_return*100:.1f}%"
            )
            new_status = RiskStatus.RED

        if self.consecutive_losses >= cb_cfg.get("consecutive_losses", 5):
            alerts.append(
                f"{self.consecutive_losses} PERTES CONSÉCUTIVES → Pause forcée"
            )
            new_status = RiskStatus.RED

        # ── Vérifications YELLOW (si pas déjà RED) ──
        if new_status != RiskStatus.RED:
            yellow_dd_threshold = -self.cfg["max_drawdown_pct"] * 0.60
            if current_drawdown_pct <= yellow_dd_threshold:
                alerts.append(
                    f"Alerte drawdown : {current_drawdown_pct:.1f}% "
                    f"(seuil alerte: {yellow_dd_threshold:.1f}%)"
                )
                new_status = RiskStatus.YELLOW

            yellow_loss_threshold = -self.cfg["max_daily_loss_pct"] * 0.60
            if daily_loss_pct <= yellow_loss_threshold:
                alerts.append(f"Alerte perte journalière : {daily_loss_pct:.1f}%")
                new_status = RiskStatus.YELLOW

        # Mise à jour du statut global
        if new_status != self.status:
            logger.warning(
                f"CHANGEMENT DE STATUT : {self.status.value} → {new_status.value}"
            )
            if new_status == RiskStatus.RED:
                for alert in alerts:
                    logger.error(f"  ⛔ {alert}")
            elif new_status == RiskStatus.YELLOW:
                for alert in alerts:
                    logger.warning(f"  ⚠  {alert}")

        self.status = new_status
        return new_status, alerts

    # ─────────────────────────────────────────────────────────────
    #  ÉVALUATION COMPLÈTE DU PORTEFEUILLE
    # ─────────────────────────────────────────────────────────────

    def evaluate_portfolio(
        self,
        portfolio_values: pd.Series,
        positions: Dict[str, Dict],
        daily_return: float = 0.0,
        market_return: float = 0.0
    ) -> PortfolioRisk:
        """
        Évalue le risque global du portefeuille.

        PARAMÈTRES :
        - portfolio_values : Série historique des valeurs du portefeuille
        - positions        : Dict {symbol: {"value": $, "entry_price": $}}
        - daily_return     : Rendement du portefeuille aujourd'hui
        - market_return    : Rendement de l'indice de référence aujourd'hui

        RETOURNE :
        - PortfolioRisk : Rapport complet du risque
        """
        portfolio_val = float(portfolio_values.iloc[-1]) if len(portfolio_values) > 0 else 0

        # ── Drawdown ──
        current_dd, max_dd = self.compute_drawdown(portfolio_values)

        # ── VaR ──
        returns = portfolio_values.pct_change().dropna()
        var_95 = self.compute_var(returns, confidence=0.95)

        # ── Exposition totale ──
        total_invested = sum(p.get("value", 0) for p in positions.values())
        exposure_pct = total_invested / (portfolio_val + 1e-8) * 100

        # ── Corrélation (simplifiée) ──
        n_pos = len(positions)
        corr_risk = min(n_pos / 10.0, 1.0)  # Proxy simple

        # ── Circuit breakers ──
        status, alerts = self.check_circuit_breakers(
            current_drawdown_pct=current_dd,
            daily_loss_pct=daily_return * 100,
            market_daily_return=market_return
        )

        # ── Multiplicateur de position ──
        if status == RiskStatus.RED:
            pos_mult = 0.0
        elif status == RiskStatus.YELLOW:
            pos_mult = 0.5
        else:
            pos_mult = 1.0

        # ── Rapport par position ──
        position_risks = []
        for sym, pos in positions.items():
            entry = pos.get("entry_price", 0)
            current = pos.get("current_price", entry)
            val = pos.get("value", 0)
            pnl = (current - entry) / (entry + 1e-8) * 100 if entry > 0 else 0
            size_pct = val / (portfolio_val + 1e-8) * 100

            pos_alerts = []
            if pnl <= -self.cfg["stop_loss_pct"] * 0.7:
                pos_alerts.append(f"Proche du stop-loss ({pnl:.1f}%)")
            if size_pct > self.cfg["max_position_size_pct"]:
                pos_alerts.append(f"Taille excessive ({size_pct:.1f}%)")

            risk_level = "LOW"
            if abs(pnl) > 5 or size_pct > 7:
                risk_level = "MEDIUM"
            if abs(pnl) > 8 or size_pct > 9:
                risk_level = "HIGH"

            position_risks.append(PositionRisk(
                symbol=sym, size_pct=size_pct, unrealized_pnl=pnl,
                stop_loss_pct=-self.cfg["stop_loss_pct"],
                risk_level=risk_level, alerts=pos_alerts
            ))

        return PortfolioRisk(
            total_exposure_pct=exposure_pct,
            current_drawdown=current_dd,
            daily_var=var_95,
            correlation_risk=corr_risk,
            status=status,
            alerts=alerts,
            position_risks=position_risks,
            can_trade=(status != RiskStatus.RED),
            position_multiplier=pos_mult
        )

    def reset_circuit_breakers(self, manual: bool = False):
        """
        Réinitialise les circuit breakers (reset manuel ou automatique).

        À UTILISER AVEC EXTRÊME PRUDENCE.
        Un reset manuel doit être décidé par un humain après analyse.
        """
        if manual:
            self.consecutive_losses = 0
            self.status = RiskStatus.GREEN
            logger.info("Circuit breakers réinitialisés MANUELLEMENT ✓")
        else:
            # Reset automatique seulement si la situation s'est améliorée
            logger.info(
                "Reset automatique refusé. "
                "Utilisez reset_circuit_breakers(manual=True) après analyse."
            )
