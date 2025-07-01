import logging
import random
from abc import abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
from pandas import DataFrame
from freqtrade.exceptions import OperationalException

logger = logging.getLogger(__name__)

class BaseActions(Enum):
    """
    Default action space, mostly used for type handling.
    """
    Neutral = 0
    Long_enter = 1
    Long_exit = 2
    Short_enter = 3
    Short_exit = 4

class Positions(Enum):
    Short = 0
    Long = 1
    Neutral = 0.5

    def opposite(self) -> 'Positions':
        return Positions.Short if self == Positions.Long else Positions.Long

class BaseEnvironment(gym.Env):
    """
    Base class for environments. This class is agnostic to action count.
    Inherited classes customize this to include varying action counts/types,
    See RL/Base5ActionRLEnv.py and RL/Base4ActionRLEnv.py
    """

    def __init__(
        self,
        df: DataFrame = DataFrame(),
        prices: DataFrame = DataFrame(),
        reward_kwargs: Dict[str, Any] = {},
        window_size: int = 10,
        starting_point: bool = True,
        id: str = 'baseenv-1',
        seed: int = 1,
        config: Dict[str, Any] = {},
        live: bool = False,
        fee: float = 0.0015,
        can_short: bool = False,
        pair: str = '',
        df_raw: DataFrame = DataFrame()
    ) -> None:
        self.config = config
        self.rl_config = config['freqai']['rl_config']
        self.add_state_info = self.rl_config.get('add_state_info', False)
        self.id = id
        self.max_drawdown = 1 - self.rl_config.get('max_training_drawdown_pct', 0.8)
        self.compound_trades = config['stake_amount'] == 'unlimited'
        self.pair = pair
        self.raw_features = df_raw
        if self.config.get('fee', None) is not None:
            self.fee = self.config['fee']
        else:
            self.fee = fee
        self.actions = BaseActions
        self.tensorboard_metrics: Dict[str, Dict[str, Any]] = {}
        self.can_short = can_short
        self.live = live
        if not self.live and self.add_state_info:
            raise OperationalException('`add_state_info` is not available in backtesting. Change parameter to false in your rl_config. See `add_state_info` docs for more info.')
        self.seed(seed)
        self.reset_env(df, prices, window_size, reward_kwargs, starting_point)

    def reset_env(
        self,
        df: DataFrame,
        prices: DataFrame,
        window_size: int,
        reward_kwargs: Dict[str, Any],
        starting_point: bool = True
    ) -> None:
        self.signal_features = df
        self.prices = prices
        self.window_size = window_size
        self.starting_point = starting_point
        self.rr = reward_kwargs['rr']
        self.profit_aim = reward_kwargs['profit_aim']
        if self.add_state_info:
            self.total_features = self.signal_features.shape[1] + 3
        else:
            self.total_features = self.signal_features.shape[1]
        self.shape = (window_size, self.total_features)
        self.set_action_space()
        self.observation_space = spaces.Box(low=-1, high=1, shape=self.shape, dtype=np.float32)
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick: Optional[int] = None
        self._position = Positions.Neutral
        self._position_history: List[Optional[Positions]] = [None]
        self.total_reward = 0.0
        self._total_profit = 1.0
        self._total_unrealized_profit = 1.0
        self.history: Dict[str, List[Any]] = {}
        self.trade_history: List[Any] = []

    def get_attr(self, attr: str) -> Any:
        """
        Returns the attribute of the environment
        :param attr: attribute to return
        :return: attribute
        """
        return getattr(self, attr)

    @abstractmethod
    def set_action_space(self) -> None:
        """
        Unique to the environment action count. Must be inherited.
        """

    def action_masks(self) -> List[bool]:
        return [self._is_valid(action.value) for action in self.actions]

    def seed(self, seed: int = 1) -> List[int]:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def tensorboard_log(
        self,
        metric: str,
        value: Optional[float] = None,
        inc: Optional[bool] = None,
        category: str = 'custom'
    ) -> None:
        increment = True if value is None else False
        value = 1 if increment else value
        if category not in self.tensorboard_metrics:
            self.tensorboard_metrics[category] = {}
        if not increment or metric not in self.tensorboard_metrics[category]:
            self.tensorboard_metrics[category][metric] = value
        else:
            self.tensorboard_metrics[category][metric] += value

    def reset_tensorboard_log(self) -> None:
        self.tensorboard_metrics = {}

    def reset(self, seed: Optional[int] = None) -> Tuple[DataFrame, Dict[str, List[Any]]]:
        """
        Reset is called at the beginning of every episode
        """
        self.reset_tensorboard_log()
        self._done = False
        if self.starting_point is True:
            if self.rl_config.get('randomize_starting_position', False):
                length_of_data = int(self._end_tick / 4)
                start_tick = random.randint(self.window_size + 1, length_of_data)
                self._start_tick = start_tick
            self._position_history = self._start_tick * [None] + [self._position]
        else:
            self._position_history = self.window_size * [None] + [self._position]
        self._current_tick = self._start_tick
        self._last_trade_tick = None
        self._position = Positions.Neutral
        self.total_reward = 0.0
        self._total_profit = 1.0
        self.history = {}
        self.trade_history = []
        self.portfolio_log_returns = np.zeros(len(self.prices))
        self._profits: List[Tuple[int, float]] = [(self._start_tick, 1)]
        self.close_trade_profit: List[float] = []
        self._total_unrealized_profit = 1.0
        return (self._get_observation(), self.history)

    @abstractmethod
    def step(self, action: int) -> Tuple[DataFrame, float, bool, Dict[str, Any]]:
        """
        Step depends on action types, this must be inherited.
        """
        return

    def _get_observation(self) -> DataFrame:
        """
        This may or may not be independent of action types, user can inherit
        this in their custom "MyRLEnv"
        """
        features_window = self.signal_features[self._current_tick - self.window_size:self._current_tick]
        if self.add_state_info:
            features_and_state = DataFrame(np.zeros((len(features_window), 3)), columns=['current_profit_pct', 'position', 'trade_duration'], index=features_window.index)
            features_and_state['current_profit_pct'] = self.get_unrealized_profit()
            features_and_state['position'] = self._position.value
            features_and_state['trade_duration'] = self.get_trade_duration()
            features_and_state = pd.concat([features_window, features_and_state], axis=1)
            return features_and_state
        else:
            return features_window

    def get_trade_duration(self) -> int:
        """
        Get the trade duration if the agent is in a trade
        """
        if self._last_trade_tick is None:
            return 0
        else:
            return self._current_tick - self._last_trade_tick

    def get_unrealized_profit(self) -> float:
        """
        Get the unrealized profit if the agent is in a trade
        """
        if self._last_trade_tick is None:
            return 0.0
        if self._position == Positions.Neutral:
            return 0.0
        elif self._position == Positions.Short:
            current_price = self.add_entry_fee(self.prices.iloc[self._current_tick].open)
            last_trade_price = self.add_exit_fee(self.prices.iloc[self._last_trade_tick].open)
            return (last_trade_price - current_price) / last_trade_price
        elif self._position == Positions.Long:
            current_price = self.add_exit_fee(self.prices.iloc[self._current_tick].open)
            last_trade_price = self.add_entry_fee(self.prices.iloc[self._last_trade_tick].open)
            return (current_price - last_trade_price) / last_trade_price
        else:
            return 0.0

    @abstractmethod
    def is_tradesignal(self, action: int) -> bool:
        """
        Determine if the signal is a trade signal. This is
        unique to the actions in the environment, and therefore must be
        inherited.
        """
        return True

    def _is_valid(self, action: int) -> bool:
        """
        Determine if the signal is valid.This is
        unique to the actions in the environment, and therefore must be
        inherited.
        """
        return True

    def add_entry_fee(self, price: float) -> float:
        return price * (1 + self.fee)

    def add_exit_fee(self, price: float) -> float:
        return price / (1 + self.fee)

    def _update_history(self, info: Dict[str, Any]) -> None:
        if not self.history:
            self.history = {key: [] for key in info.keys()}
        for key, value in info.items():
            self.history[key].append(value)

    @abstractmethod
    def calculate_reward(self, action: int) -> float:
        """
        An example reward function. This is the one function that users will likely
        wish to inject their own creativity into.

        Warning!
        This is function is a showcase of functionality designed to show as many possible
        environment control features as possible. It is also designed to run quickly
        on small computers. This is a benchmark, it is *not* for live production.

        :param action: int = The action made by the agent for the current candle.
        :return:
        float = the reward to give to the agent for current step (used for optimization
            of weights in NN)
        """

    def _update_unrealized_total_profit(self) -> None:
        """
        Update the unrealized total profit in case of episode end.
        """
        if self._position in (Positions.Long, Positions.Short):
            pnl = self.get_unrealized_profit()
            if self.compound_trades:
                unrl_profit = self._total_profit * (1 + pnl)
            else:
                unrl_profit = self._total_profit + pnl
            self._total_unrealized_profit = unrl_profit

    def _update_total_profit(self) -> None:
        pnl = self.get_unrealized_profit()
        if self.compound_trades:
            self._total_profit = self._total_profit * (1 + pnl)
        else:
            self._total_profit += pnl

    def current_price(self) -> float:
        return self.prices.iloc[self._current_tick].open

    def get_actions(self) -> Enum:
        """
        Used by SubprocVecEnv to get actions from
        initialized env for tensorboard callback
        """
        return self.actions
