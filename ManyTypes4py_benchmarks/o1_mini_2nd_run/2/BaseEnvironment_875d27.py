import logging
import random
from abc import ABC, abstractmethod
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


class BaseEnvironment(gym.Env, ABC):
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
        """
        Initializes the training/eval environment.
        :param df: dataframe of features
        :param prices: dataframe of prices to be used in the training environment
        :param window_size: size of window (temporal) to pass to the agent
        :param reward_kwargs: extra config settings assigned by user in `rl_config`
        :param starting_point: start at edge of window or not
        :param id: string id of the environment (used in backend for multiprocessed env)
        :param seed: Sets the seed of the environment higher in the gym.Env object
        :param config: Typical user configuration file
        :param live: Whether or not this environment is active in dry/live/backtesting
        :param fee: The fee to use for environmental interactions.
        :param can_short: Whether or not the environment can short
        """
        self.config: Dict[str, Any] = config
        self.rl_config: Dict[str, Any] = config.get('freqai', {}).get('rl_config', {})
        self.add_state_info: bool = self.rl_config.get('add_state_info', False)
        self.id: str = id
        self.max_drawdown: float = 1 - self.rl_config.get('max_training_drawdown_pct', 0.8)
        self.compound_trades: bool = config.get('stake_amount') == 'unlimited'
        self.pair: str = pair
        self.raw_features: DataFrame = df_raw
        self.fee: float = self.config.get('fee', fee)
        self.actions: Enum = BaseActions
        self.tensorboard_metrics: Dict[str, Dict[str, float]] = {}
        self.can_short: bool = can_short
        self.live: bool = live
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
        """
        Resets the environment when the agent fails (in our case, if the drawdown
        exceeds the user set max_training_drawdown_pct)
        :param df: dataframe of features
        :param prices: dataframe of prices to be used in the training environment
        :param window_size: size of window (temporal) to pass to the agent
        :param reward_kwargs: extra config settings assigned by user in `rl_config`
        :param starting_point: start at edge of window or not
        """
        self.signal_features: DataFrame = df
        self.prices: DataFrame = prices
        self.window_size: int = window_size
        self.starting_point: bool = starting_point
        self.rr: Any = reward_kwargs.get('rr')
        self.profit_aim: Any = reward_kwargs.get('profit_aim')
        self.total_features: int = self.signal_features.shape[1] + 3 if self.add_state_info else self.signal_features.shape[1]
        self.shape: Tuple[int, int] = (window_size, self.total_features)
        self.set_action_space()
        self.observation_space: spaces.Box = spaces.Box(low=-1, high=1, shape=self.shape, dtype=np.float32)
        self._start_tick: int = self.window_size
        self._end_tick: int = len(self.prices) - 1
        self._done: bool = False
        self._current_tick: int = self._start_tick
        self._last_trade_tick: Optional[int] = None
        self._position: Positions = Positions.Neutral
        self._position_history: List[Optional[Positions]] = [None]
        self.total_reward: float = 0.0
        self._total_profit: float = 1.0
        self._total_unrealized_profit: float = 1.0
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
        pass

    def action_masks(self) -> List[bool]:
        return [self._is_valid(action.value) for action in self.actions]

    def seed(self, seed: int = 1) -> List[int]:
        self.np_random, seed_out = seeding.np_random(seed)
        return [seed_out]

    def tensorboard_log(
        self,
        metric: str,
        value: Optional[float] = None,
        inc: Optional[bool] = None,
        category: str = 'custom'
    ) -> None:
        """
        Function builds the tensorboard_metrics dictionary
        to be parsed by the TensorboardCallback. This
        function is designed for tracking incremented objects,
        events, actions inside the training environment.
        For example, a user can call this to track the
        frequency of occurrence of an `is_valid` call in
        their `calculate_reward()`:

        def calculate_reward(self, action: int) -> float:
            if not self._is_valid(action):
                self.tensorboard_log("invalid")
                return -2

        :param metric: metric to be tracked and incremented
        :param value: `metric` value
        :param inc: (deprecated) sets whether the `value` is incremented or not
        :param category: `metric` category
        """
        increment: bool = True if value is None else False
        value = 1.0 if increment else value
        if category not in self.tensorboard_metrics:
            self.tensorboard_metrics[category] = {}
        if not increment or metric not in self.tensorboard_metrics[category]:
            self.tensorboard_metrics[category][metric] = value  # type: ignore
        else:
            self.tensorboard_metrics[category][metric] += value  # type: ignore

    def reset_tensorboard_log(self) -> None:
        self.tensorboard_metrics = {}

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset is called at the beginning of every episode
        """
        self.reset_tensorboard_log()
        self._done = False
        if self.starting_point:
            if self.rl_config.get('randomize_starting_position', False):
                length_of_data: int = int(self._end_tick / 4)
                start_tick: int = random.randint(self.window_size + 1, length_of_data)
                self._start_tick = start_tick
            self._position_history = [None] * self._start_tick + [self._position]
        else:
            self._position_history = [None] * self.window_size + [self._position]
        self._current_tick = self._start_tick
        self._last_trade_tick = None
        self._position = Positions.Neutral
        self.total_reward = 0.0
        self._total_profit = 1.0
        self.history = {}
        self.trade_history = []
        self.portfolio_log_returns: np.ndarray = np.zeros(len(self.prices))
        self._profits: List[Tuple[int, float]] = [(self._start_tick, 1.0)]
        self.close_trade_profit: List[Any] = []
        self._total_unrealized_profit = 1.0
        return self._get_observation(), self.history

    @abstractmethod
    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Step depends on action types, this must be inherited.
        """
        pass

    def _get_observation(self) -> Union[DataFrame, pd.DataFrame]:
        """
        This may or may not be independent of action types, user can inherit
        this in their custom "MyRLEnv"
        """
        features_window: DataFrame = self.signal_features.iloc[self._current_tick - self.window_size:self._current_tick]
        if self.add_state_info:
            features_and_state: DataFrame = DataFrame(
                np.zeros((len(features_window), 3)),
                columns=['current_profit_pct', 'position', 'trade_duration'],
                index=features_window.index
            )
            features_and_state['current_profit_pct'] = self.get_unrealized_profit()
            features_and_state['position'] = self._position.value
            features_and_state['trade_duration'] = self.get_trade_duration()
            features_and_state = pd.concat([features_window, features_and_state], axis=1)
            return features_and_state
        else:
            return features_window

    def get_trade_duration(self) -> float:
        """
        Get the trade duration if the agent is in a trade
        """
        if self._last_trade_tick is None:
            return 0.0
        else:
            return float(self._current_tick - self._last_trade_tick)

    def get_unrealized_profit(self) -> float:
        """
        Get the unrealized profit if the agent is in a trade
        """
        if self._last_trade_tick is None:
            return 0.0
        if self._position == Positions.Neutral:
            return 0.0
        elif self._position == Positions.Short:
            current_price: float = self.add_entry_fee(self.prices.iloc[self._current_tick].open)
            last_trade_price: float = self.add_exit_fee(self.prices.iloc[self._last_trade_tick].open)
            return (last_trade_price - current_price) / last_trade_price
        elif self._position == Positions.Long:
            current_price: float = self.add_exit_fee(self.prices.iloc[self._current_tick].open)
            last_trade_price: float = self.add_entry_fee(self.prices.iloc[self._last_trade_tick].open)
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
        pass

    @abstractmethod
    def _is_valid(self, action: int) -> bool:
        """
        Determine if the signal is valid. This is
        unique to the actions in the environment, and therefore must be
        inherited.
        """
        pass

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
        pass

    def _update_unrealized_total_profit(self) -> None:
        """
        Update the unrealized total profit in case of episode end.
        """
        if self._position in (Positions.Long, Positions.Short):
            pnl: float = self.get_unrealized_profit()
            if self.compound_trades:
                unrl_profit: float = self._total_profit * (1 + pnl)
            else:
                unrl_profit: float = self._total_profit + pnl
            self._total_unrealized_profit = unrl_profit

    def _update_total_profit(self) -> None:
        pnl: float = self.get_unrealized_profit()
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
