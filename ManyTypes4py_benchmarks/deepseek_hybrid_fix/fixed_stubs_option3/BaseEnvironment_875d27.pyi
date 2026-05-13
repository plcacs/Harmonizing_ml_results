from abc import abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from pandas import DataFrame

logger: logging.Logger

class BaseActions(Enum):
    Neutral = 0
    Long_enter = 1
    Long_exit = 2
    Short_enter = 3
    Short_exit = 4

class Positions(Enum):
    Short = 0
    Long = 1
    Neutral = 0.5

    def opposite(self) -> 'Positions': ...

class BaseEnvironment(gym.Env):
    config: Dict[str, Any]
    rl_config: Dict[str, Any]
    add_state_info: bool
    id: str
    max_drawdown: float
    compound_trades: bool
    pair: str
    raw_features: DataFrame
    fee: float
    actions: Type[BaseActions]
    can_short: bool
    live: bool
    signal_features: DataFrame
    prices: DataFrame
    window_size: int
    starting_point: bool
    rr: float
    profit_aim: float
    total_features: int
    shape: Tuple[int, int]
    observation_space: spaces.Box
    _start_tick: int
    _end_tick: int
    _done: bool
    _current_tick: int
    _last_trade_tick: Optional[int]
    _position: Positions
    total_reward: float
    _total_profit: float
    _total_unrealized_profit: float
    portfolio_log_returns: np.ndarray
    _profits: List[Tuple[int, float]]

    def __init__(
        self,
        df: DataFrame = ...,
        prices: DataFrame = ...,
        reward_kwargs: Dict[str, Any] = ...,
        window_size: int = ...,
        starting_point: bool = ...,
        id: str = ...,
        seed: int = ...,
        config: Dict[str, Any] = ...,
        live: bool = ...,
        fee: float = ...,
        can_short: bool = ...,
        pair: str = ...,
        df_raw: DataFrame = ...
    ) -> None: ...

    def reset_env(
        self,
        df: DataFrame,
        prices: DataFrame,
        window_size: int,
        reward_kwargs: Dict[str, Any],
        starting_point: bool = ...
    ) -> None: ...

    def get_attr(self, attr: str) -> Any: ...

    @abstractmethod
    def set_action_space(self) -> None: ...

    def action_masks(self) -> List[bool]: ...

    def seed(self, seed: int = ...) -> List[int]: ...

    def tensorboard_log(
        self,
        metric: str,
        value: Optional[float] = ...,
        inc: Optional[Any] = ...,
        category: str = ...
    ) -> None: ...

    def reset_tensorboard_log(self) -> None: ...

    def reset(self, seed: Optional[int] = ...) -> Tuple[pd.DataFrame, Dict[str, List[Any]]]: ...

    @abstractmethod
    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict[str, Any]]: ...

    def _get_observation(self) -> pd.DataFrame: ...

    def get_trade_duration(self) -> int: ...

    def get_unrealized_profit(self) -> float: ...

    @abstractmethod
    def is_tradesignal(self, action: int) -> bool: ...

    def _is_valid(self, action: int) -> bool: ...

    def add_entry_fee(self, price: float) -> float: ...

    def add_exit_fee(self, price: float) -> float: ...

    def _update_history(self, info: Dict[str, Any]) -> None: ...

    @abstractmethod
    def calculate_reward(self, action: int) -> float: ...

    def _update_unrealized_total_profit(self) -> None: ...

    def _update_total_profit(self) -> None: ...

    def current_price(self) -> float: ...

    def get_actions(self) -> Type[BaseActions]: ...