"""
AGENT DE REINFORCEMENT LEARNING  (Couche 5)
=============================================

QU'EST-CE QUE LE REINFORCEMENT LEARNING (RL) ?
Le RL est l'apprentissage par essai-erreur et récompenses.

ANALOGIE :
C'est comme entraîner un chien : quand il fait quelque chose
de bien (profit), on le récompense (+). Quand il fait mal
(perte), on le punit (-). Avec le temps, il apprend à
maximiser les récompenses.

DANS LE TRADING :
- L'AGENT = le système de trading
- L'ENVIRONNEMENT = le marché financier simulé
- L'ÉTAT = les features du marché à l'instant t
- L'ACTION = Acheter / Vendre / Ne rien faire
- LA RÉCOMPENSE = Profit ajusté au risque (Sharpe)

ALGORITHME : PPO (Proximal Policy Optimization)
PPO est l'un des algorithmes RL les plus populaires et stables.
Il est utilisé par OpenAI pour entraîner des systèmes complexes.
Son avantage : il évite les mises à jour trop brutales de la
politique, ce qui stabilise l'entraînement.

CYCLE D'APPRENTISSAGE :
1. L'agent observe l'état du marché
2. Il choisit une action (achat/vente/hold)
3. Le marché évolue → nouveau profit/perte
4. L'agent reçoit une récompense (Sharpe incrémental)
5. Il met à jour sa politique pour maximiser les récompenses futures
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, List
import torch

from ..utils.logger import get_logger

logger = get_logger("RLAgent")


# ══════════════════════════════════════════════════════════════════
#  ENVIRONNEMENT DE TRADING (simulation du marché)
# ══════════════════════════════════════════════════════════════════

class TradingEnvironment(gym.Env):
    """
    Environnement de trading compatible avec Gymnasium (OpenAI Gym).

    QU'EST-CE QU'UN ENVIRONNEMENT GYM ?
    C'est une interface standard pour les problèmes de RL.
    Elle définit :
    - reset()  : Réinitialise l'environnement au début d'un épisode
    - step()   : Exécute une action, retourne (état, récompense, terminé)
    - render() : Affiche l'état (optionnel)

    UN ÉPISODE = simulation complète sur les données historiques.

    ESPACE D'ACTIONS (DISCRET) :
    0 = HOLD   (ne rien faire)
    1 = BUY    (acheter avec 20% du capital disponible)
    2 = SELL   (vendre toutes les positions)

    ESPACE D'ÉTATS :
    Vecteur de features du marché + état du portefeuille :
    [features_marché..., position_courante, profit_latent, drawdown]
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df_features: pd.DataFrame,
        prices: pd.Series,
        config: dict,
        initial_capital: float = 100_000.0
    ):
        """
        PARAMÈTRES :
        - df_features    : DataFrame des features normalisées
        - prices         : Série des prix de clôture réels
        - config         : Configuration globale
        - initial_capital: Capital de départ en USD
        """
        super().__init__()

        self.features = df_features.values.astype(np.float32)
        self.prices   = prices.values.astype(np.float32)
        self.config   = config
        self.initial_capital = initial_capital

        # Vérification de cohérence
        assert len(self.features) == len(self.prices), \
            f"Dimensions incohérentes: features={len(self.features)}, prices={len(self.prices)}"

        self.n_steps   = len(self.features)
        self.n_features = self.features.shape[1]

        # Paramètres de simulation
        env_cfg = config["rl_agent"]["environment"]
        self.transaction_cost = env_cfg["transaction_cost"]
        self.max_position_pct = env_cfg["max_position_size"]

        # ── Espaces Gymnasium ──
        # Actions : 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)

        # État : features + [position%, profit%, drawdown%]
        n_state = self.n_features + 3
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(n_state,), dtype=np.float32
        )

        # État interne (initialisé par reset())
        self.current_step     = 0
        self.capital          = initial_capital
        self.shares_held      = 0.0
        self.peak_portfolio   = initial_capital
        self.trade_history: List[Dict] = []

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Réinitialise l'environnement au début d'un épisode.

        RETOURNE :
        - observation : État initial
        - info        : Informations supplémentaires (dict vide)
        """
        super().reset(seed=seed)

        self.current_step   = 0
        self.capital        = self.initial_capital
        self.shares_held    = 0.0
        self.peak_portfolio = self.initial_capital
        self.trade_history  = []

        return self._get_observation(), {}

    def _portfolio_value(self) -> float:
        """Valeur totale du portefeuille (cash + actions)."""
        return self.capital + self.shares_held * float(self.prices[self.current_step])

    def _get_observation(self) -> np.ndarray:
        """
        Construit le vecteur d'état de l'agent.

        ÉTAT = [features_marché, position%, profit%, drawdown%]

        - position%  : Quel % du portefeuille est en actions (0.0 → 1.0)
        - profit%    : Profit/perte depuis le début de l'épisode
        - drawdown%  : Drawdown actuel depuis le sommet
        """
        market_features = self.features[self.current_step]

        portfolio_val = self._portfolio_value()

        # Position en % du portefeuille
        stock_value = self.shares_held * float(self.prices[self.current_step])
        position_pct = stock_value / (portfolio_val + 1e-8)

        # Profit total
        profit_pct = (portfolio_val - self.initial_capital) / self.initial_capital

        # Drawdown courant
        self.peak_portfolio = max(self.peak_portfolio, portfolio_val)
        drawdown = (portfolio_val - self.peak_portfolio) / (self.peak_portfolio + 1e-8)

        portfolio_features = np.array(
            [position_pct, profit_pct, drawdown], dtype=np.float32
        )

        return np.concatenate([market_features, portfolio_features])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Exécute une action et avance d'un pas dans le temps.

        PARAMÈTRE :
        - action : 0=Hold, 1=Buy, 2=Sell

        RETOURNE :
        - observation  : Nouvel état
        - reward       : Récompense (Sharpe incrémental)
        - terminated   : True si fin des données
        - truncated    : True si limite atteinte (non utilisé)
        - info         : Informations de débogage

        PROCESSUS :
        1. Exécuter l'action (acheter/vendre)
        2. Avancer d'un pas dans le temps
        3. Observer le nouveau prix
        4. Calculer la récompense
        5. Retourner le nouvel état
        """
        current_price = float(self.prices[self.current_step])

        # ── Exécution de l'action ──
        if action == 1:  # BUY
            # Acheter avec max_position_pct% du capital disponible
            amount_to_invest = self.capital * self.max_position_pct
            if amount_to_invest > 0 and self.capital > 10:
                cost = amount_to_invest * (1 + self.transaction_cost)
                if cost <= self.capital:
                    shares_bought = amount_to_invest / current_price
                    self.shares_held += shares_bought
                    self.capital -= cost
                    self.trade_history.append({
                        "step": self.current_step, "action": "BUY",
                        "price": current_price, "shares": shares_bought
                    })

        elif action == 2:  # SELL
            if self.shares_held > 0:
                proceeds = self.shares_held * current_price
                self.capital += proceeds * (1 - self.transaction_cost)
                self.trade_history.append({
                    "step": self.current_step, "action": "SELL",
                    "price": current_price, "shares": self.shares_held
                })
                self.shares_held = 0.0

        # ── Avancer dans le temps ──
        self.current_step += 1
        terminated = self.current_step >= self.n_steps - 1

        # ── Calculer la récompense ──
        new_portfolio_val = self._portfolio_value()
        prev_portfolio_val = new_portfolio_val  # sera mis à jour ci-dessous

        # Rendement de cette période
        if self.current_step > 0:
            prev_price = float(self.prices[self.current_step - 1])
            prev_val = self.capital + self.shares_held * prev_price
            step_return = (new_portfolio_val - prev_val) / (prev_val + 1e-8)
        else:
            step_return = 0.0

        # Récompense = rendement ajusté au risque
        reward = self._compute_reward(step_return)

        # Pénaliser si drawdown trop important (circuit breaker)
        drawdown = (new_portfolio_val - self.peak_portfolio) / (self.peak_portfolio + 1e-8)
        if drawdown < -0.15:  # -15% de drawdown max
            reward -= 1.0     # Pénalité forte

        info = {
            "portfolio_value": new_portfolio_val,
            "step_return"    : step_return,
            "drawdown"       : drawdown,
            "n_trades"       : len(self.trade_history)
        }

        return self._get_observation(), float(reward), terminated, False, info

    def _compute_reward(self, step_return: float) -> float:
        """
        Calcule la récompense de trading.

        FONCTION DE RÉCOMPENSE :
        On utilise le rendement ajusté à la volatilité récente.
        C'est une approximation du Sharpe ratio incrémental.

        reward = rendement / volatilité_récente

        POURQUOI PAS JUSTE LE RENDEMENT ?
        Un rendement de +5% avec une volatilité de 20% est
        bien moins impressionnant qu'un +5% avec 2% de volatilité.
        On veut récompenser l'EFFICACITÉ, pas juste le profit brut.
        """
        # Simple : récompense directe + bonus pour gestion du risque
        reward = step_return * 100  # Convertir en %

        # Bonus si rendement positif avec faible drawdown
        if step_return > 0:
            reward *= 1.5

        return float(np.clip(reward, -10.0, 10.0))


# ══════════════════════════════════════════════════════════════════
#  AGENT RL
# ══════════════════════════════════════════════════════════════════

class RLAgent:
    """
    Wrapper autour de Stable-Baselines3 pour l'agent PPO.

    POURQUOI STABLE-BASELINES3 ?
    C'est une bibliothèque Python de référence pour le RL,
    maintenue par des chercheurs. Elle implémente PPO, SAC, DQN
    avec des optimisations robustes et testées.

    UTILISATION :
        agent = RLAgent(config)
        agent.train(env_train)
        action = agent.predict(observation)
    """

    def __init__(self, config: dict):
        self.config = config
        self.cfg    = config["rl_agent"]
        self.model  = None

    def build(self, env: TradingEnvironment) -> "RLAgent":
        """
        Construit l'agent PPO avec l'environnement donné.

        QU'EST-CE QUE PPO ?
        PPO = Proximal Policy Optimization (OpenAI, 2017)

        PARAMÈTRES CLÉS :
        - learning_rate : Vitesse d'adaptation (0.0003 = standard)
        - n_steps       : Pas collectés avant chaque mise à jour
        - gamma         : Discount factor (0.99 = futur proche/lointain équilibrés)
        - clip_range    : Limite les mises à jour (0.2 = ±20% max)

        POURQUOI PPO ET PAS DQN OU SAC ?
        - DQN : Bon pour actions discrètes simples, mais moins stable
        - SAC : Excellent pour actions continues, mais nos actions sont discrètes
        - PPO : Stable, général, marche bien pour le trading discret ✓
        """
        try:
            from stable_baselines3 import PPO, SAC, DQN
            from stable_baselines3.common.vec_env import DummyVecEnv

            algo = self.cfg["algorithm"].upper()
            ppo_cfg = self.cfg["ppo"]

            wrapped_env = DummyVecEnv([lambda: env])

            if algo == "PPO":
                self.model = PPO(
                    policy="MlpPolicy",
                    env=wrapped_env,
                    learning_rate=ppo_cfg["learning_rate"],
                    n_steps=ppo_cfg["n_steps"],
                    batch_size=ppo_cfg["batch_size"],
                    n_epochs=ppo_cfg["n_epochs"],
                    gamma=ppo_cfg["gamma"],
                    gae_lambda=ppo_cfg["gae_lambda"],
                    clip_range=ppo_cfg["clip_range"],
                    verbose=0,
                    seed=self.config["system"]["random_seed"],
                    tensorboard_log="./logs/rl_tensorboard/"
                )
            else:
                raise ValueError(f"Algorithme '{algo}' non supporté. Utilisez 'ppo'.")

            logger.info(f"Agent {algo} construit ✓")

        except ImportError:
            logger.error(
                "stable-baselines3 non installé.\n"
                "Installez avec : pip install stable-baselines3"
            )
            raise

        return self

    def train(
        self,
        env: TradingEnvironment,
        total_timesteps: Optional[int] = None
    ) -> "RLAgent":
        """
        Entraîne l'agent RL sur l'environnement de trading.

        PARAMÈTRES :
        - env             : Environnement de trading
        - total_timesteps : Nombre de pas d'entraînement total

        RETOURNE : self (pour chaînage)
        """
        if self.model is None:
            self.build(env)

        timesteps = total_timesteps or self.cfg["training"]["total_timesteps"]
        logger.info(f"Entraînement de l'agent RL sur {timesteps:,} pas…")

        self.model.learn(
            total_timesteps=timesteps,
            progress_bar=True
        )

        logger.info("Entraînement RL terminé ✓")
        return self

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> int:
        """
        Prédit la meilleure action pour un état donné.

        PARAMÈTRES :
        - observation  : Vecteur d'état (features + portfolio)
        - deterministic: True = toujours l'action optimale
                         False = exploration aléatoire (pour entraînement)

        RETOURNE : Action (0=Hold, 1=Buy, 2=Sell)
        """
        if self.model is None:
            raise RuntimeError("L'agent n'est pas entraîné. Appelez .train() d'abord.")

        action, _ = self.model.predict(observation, deterministic=deterministic)
        return int(action)

    def save(self, path: str = "./models/rl_agent"):
        """Sauvegarde l'agent entraîné."""
        if self.model:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.model.save(path)
            logger.info(f"Agent RL sauvegardé : {path}")

    def load(self, path: str = "./models/rl_agent") -> "RLAgent":
        """Charge un agent sauvegardé."""
        try:
            from stable_baselines3 import PPO
            self.model = PPO.load(path)
            logger.info(f"Agent RL chargé : {path}")
        except Exception as e:
            logger.error(f"Impossible de charger l'agent : {e}")
        return self
