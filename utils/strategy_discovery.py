"""
Automatic strategy discovery via Reinforcement Learning and Genetic Algorithms.
"""

import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

# BacktestEngine is imported lazily inside the class to avoid circular imports
# (backtesting → utils.logger → utils → strategy_discovery → backtesting)
from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class StrategyChromosome:
    """Represents a strategy as a set of tunable parameters."""
    params: dict[str, float]
    fitness: float = -np.inf
    sharpe: float = 0.0
    drawdown: float = 0.0
    profit_factor: float = 0.0


class GeneticStrategyOptimizer:
    """
    Genetic algorithm for strategy parameter optimization.
    Evolves a population of strategy configurations.
    """

    def __init__(
        self,
        strategy_fn: Callable,
        ohlcv: pd.DataFrame,
        param_bounds: dict[str, tuple[float, float]],
        settings: Optional[Settings] = None,
    ) -> None:
        from backtesting.engine import BacktestEngine  # lazy import — avoids circular
        self.strategy_fn = strategy_fn
        self.ohlcv = ohlcv
        self.param_bounds = param_bounds
        self.settings = settings or Settings()
        cfg = self.settings.strategy_discovery
        self.population_size = cfg.ga_population_size
        self.generations = cfg.ga_generations
        self.mutation_rate = cfg.ga_mutation_rate
        self.crossover_rate = cfg.ga_crossover_rate
        self.backtester = BacktestEngine(settings)

    def _random_individual(self) -> StrategyChromosome:
        params = {}
        for name, (lo, hi) in self.param_bounds.items():
            if isinstance(lo, int) and isinstance(hi, int):
                params[name] = float(random.randint(int(lo), int(hi)))
            else:
                params[name] = random.uniform(lo, hi)
        return StrategyChromosome(params=params)

    def _evaluate(self, individual: StrategyChromosome) -> None:
        """Evaluate fitness by running a backtest."""
        try:
            result = self.backtester.run_single(self.ohlcv, self.strategy_fn, individual.params)
            # Fitness = Sharpe penalized by drawdown
            individual.sharpe = result.sharpe_ratio
            individual.drawdown = result.max_drawdown
            individual.profit_factor = result.profit_factor
            individual.fitness = (
                result.sharpe_ratio
                - max(0, abs(result.max_drawdown) - 0.20) * 5   # Penalize high DD
                + result.profit_factor * 0.1
            )
        except Exception as e:
            individual.fitness = -100.0
            logger.debug(f"Evaluation failed: {e}")

    def _crossover(self, p1: StrategyChromosome, p2: StrategyChromosome) -> StrategyChromosome:
        child_params = {}
        for key in p1.params:
            if random.random() < 0.5:
                child_params[key] = p1.params[key]
            else:
                child_params[key] = p2.params[key]
        return StrategyChromosome(params=child_params)

    def _mutate(self, individual: StrategyChromosome) -> StrategyChromosome:
        mutated = deepcopy(individual)
        for key in mutated.params:
            if random.random() < self.mutation_rate:
                lo, hi = self.param_bounds[key]
                sigma = (hi - lo) * 0.1
                mutated.params[key] = float(np.clip(
                    mutated.params[key] + random.gauss(0, sigma), lo, hi
                ))
        return mutated

    def evolve(self) -> list[StrategyChromosome]:
        """Run genetic algorithm evolution. Returns list of elite strategies."""
        logger.info(f"GA: population={self.population_size}, generations={self.generations}")

        # Initialize population
        population = [self._random_individual() for _ in range(self.population_size)]

        for gen in range(self.generations):
            # Evaluate
            for ind in population:
                if ind.fitness == -np.inf:
                    self._evaluate(ind)

            # Sort by fitness
            population.sort(key=lambda x: x.fitness, reverse=True)
            elite_count = max(2, self.population_size // 10)
            elites = population[:elite_count]

            if (gen + 1) % 10 == 0:
                logger.info(
                    f"GA Gen {gen+1}/{self.generations} | "
                    f"Best Sharpe: {elites[0].sharpe:.3f} | "
                    f"Best Fitness: {elites[0].fitness:.3f}"
                )

            # New generation
            new_population = elites.copy()
            while len(new_population) < self.population_size:
                # Tournament selection
                t1 = random.sample(population[:self.population_size // 2], 2)
                t2 = random.sample(population[:self.population_size // 2], 2)
                p1 = max(t1, key=lambda x: x.fitness)
                p2 = max(t2, key=lambda x: x.fitness)

                if random.random() < self.crossover_rate:
                    child = self._crossover(p1, p2)
                else:
                    child = deepcopy(p1)

                child = self._mutate(child)
                child.fitness = -np.inf  # Mark for re-evaluation
                new_population.append(child)

            population = new_population

        # Final evaluation
        for ind in population:
            if ind.fitness == -np.inf:
                self._evaluate(ind)
        population.sort(key=lambda x: x.fitness, reverse=True)

        # Return strategies that pass acceptance criteria
        valid = [
            ind for ind in population[:20]
            if ind.sharpe >= self.settings.backtest.min_sharpe_ratio
            and abs(ind.drawdown) <= self.settings.backtest.max_drawdown_pct
            and ind.profit_factor >= self.settings.backtest.min_profit_factor
        ]
        logger.info(f"GA complete: {len(valid)} valid strategies discovered")
        return valid


class RLStrategyAgent:
    """
    Reinforcement Learning strategy discovery using stable-baselines3.
    Uses PPO to learn optimal trading decisions from raw features.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or Settings()
        self.cfg = self.settings.strategy_discovery
        self._model = None

    def build_environment(self, ohlcv: pd.DataFrame, features: pd.DataFrame):
        """Build a Gymnasium trading environment from OHLCV + features."""
        try:
            from gymnasium import Env, spaces

            class TradingEnv(Env):
                def __init__(env_self, ohlcv: pd.DataFrame, features: pd.DataFrame):
                    super().__init__()
                    env_self.ohlcv = ohlcv.reset_index(drop=True)
                    feature_cols = [c for c in features.columns if c not in {"open", "high", "low", "close", "volume"}]
                    env_self.features = features[feature_cols].fillna(0).reset_index(drop=True)
                    env_self.n_features = len(feature_cols)
                    env_self.action_space = spaces.Discrete(3)  # 0=hold, 1=buy, 2=sell
                    env_self.observation_space = spaces.Box(
                        low=-np.inf, high=np.inf, shape=(env_self.n_features,), dtype=np.float32
                    )
                    env_self.reset()

                def reset(env_self, **kwargs):
                    env_self.current_step = 0
                    env_self.position = 0
                    env_self.entry_price = 0.0
                    env_self.capital = 10000.0
                    env_self.equity = env_self.capital
                    return env_self._get_obs(), {}

                def _get_obs(env_self):
                    idx = min(env_self.current_step, len(env_self.features) - 1)
                    return env_self.features.iloc[idx].values.astype(np.float32)

                def step(env_self, action):
                    price = float(env_self.ohlcv.iloc[env_self.current_step]["close"])
                    reward = 0.0

                    if action == 1 and env_self.position == 0:
                        env_self.position = 1
                        env_self.entry_price = price
                    elif action == 2 and env_self.position == 1:
                        pct_return = (price - env_self.entry_price) / env_self.entry_price
                        reward = pct_return * 100  # Scale reward
                        env_self.capital *= (1 + pct_return * 0.05)
                        env_self.position = 0

                    # Holding cost
                    if env_self.position == 1:
                        unrealized = (price - env_self.entry_price) / env_self.entry_price
                        reward += unrealized * 10

                    env_self.current_step += 1
                    done = env_self.current_step >= len(env_self.ohlcv) - 1
                    return env_self._get_obs(), reward, done, False, {}

            return TradingEnv(ohlcv, features)
        except ImportError as e:
            logger.warning(f"Gymnasium not available: {e}")
            return None

    def train(self, ohlcv: pd.DataFrame, features: pd.DataFrame) -> bool:
        """Train RL agent. Returns True if successful."""
        env = self.build_environment(ohlcv, features)
        if env is None:
            return False

        try:
            from stable_baselines3 import PPO
            self._model = PPO(
                "MlpPolicy",
                env,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                learning_rate=3e-4,
                verbose=0,
            )
            self._model.learn(total_timesteps=self.cfg.rl_total_timesteps)
            logger.info("RL strategy agent trained successfully")
            return True
        except Exception as e:
            logger.warning(f"RL training failed: {e}")
            return False

    def get_signal(self, obs: np.ndarray) -> int:
        """Get action (0=hold, 1=buy, 2=sell) from the RL agent."""
        if self._model is None:
            return 0
        action, _ = self._model.predict(obs, deterministic=True)
        return int(action)


class StrategyDiscovery:
    """
    Orchestrates strategy discovery combining GA + RL.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        from backtesting.engine import BacktestEngine  # lazy import
        self.settings = settings or Settings()
        self.backtester = BacktestEngine(settings)
        self.rl_agent = RLStrategyAgent(settings)

    def discover_strategies(
        self,
        ohlcv: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
        use_rl: bool = True,
    ) -> list[dict]:
        """
        Run full strategy discovery pipeline.
        Returns list of validated strategy dicts.
        """
        discovered: list[dict] = []

        # 1. GA optimization for each built-in strategy
        built_in_strategies = {
            "rsi_mean_reversion": BacktestEngine.rsi_strategy,
            "ma_crossover": BacktestEngine.ma_crossover_strategy,
            "bb_breakout": BacktestEngine.bollinger_breakout_strategy,
        }

        for strategy_name, strategy_fn in built_in_strategies.items():
            logger.info(f"Optimizing {strategy_name} via GA…")
            param_bounds = self.settings.strategy_discovery.param_bounds
            optimizer = GeneticStrategyOptimizer(
                strategy_fn=strategy_fn,
                ohlcv=ohlcv,
                param_bounds=param_bounds,
                settings=self.settings,
            )
            valid_chromosomes = optimizer.evolve()

            for chrom in valid_chromosomes[:3]:  # Keep top 3 per strategy
                is_valid, summary = self.backtester.validate_strategy(
                    ohlcv, strategy_fn, chrom.params, strategy_name=strategy_name
                )
                if is_valid:
                    discovered.append({
                        "name": strategy_name,
                        "params": chrom.params,
                        "sharpe": chrom.sharpe,
                        "max_drawdown": chrom.drawdown,
                        "profit_factor": chrom.profit_factor,
                        "discovery_method": "genetic_algorithm",
                    })

        # 2. RL-based strategy (if features available)
        if use_rl and features is not None and not features.empty:
            logger.info("Training RL strategy agent…")
            success = self.rl_agent.train(ohlcv, features)
            if success:
                discovered.append({
                    "name": "rl_ppo_agent",
                    "params": {},
                    "sharpe": None,
                    "max_drawdown": None,
                    "profit_factor": None,
                    "discovery_method": "reinforcement_learning",
                })

        logger.info(f"Strategy discovery complete: {len(discovered)} valid strategies")
        return discovered
