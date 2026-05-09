import logging
import random
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
from pandas import DataFrame
from freqtrade.exceptions import OperationalException

logger = logging.getLogger(__name__)

class BaseActions(Enum):
    Neutral: int
    Long_enter: int
    Long_exit: int
    Short_enter: int
    Short_exit: int

class Positions(Enum):
    Short: int
    Long: int
    Neutral: float

    def opposite(self) -> 'Positions':
        ...

class BaseEnvironment(gym.Env):
    def __init__(self, df: DataFrame = ..., prices: DataFrame = ..., reward_kwargs: dict = ..., window_size: int = ..., starting_point: bool = ..., id: str = ..., seed: int = ..., config: dict = ..., live: bool = ..., fee: float = ..., can_short: bool = ..., pair: str = ..., df_raw: DataFrame = ...):
        ...

    def reset_env(self, df: DataFrame, prices: DataFrame, window_size: int, reward_kwargs: dict, starting_point: bool = ...) -> None:
        ...

    def get_attr(self, attr: str) -> Any:
        ...

    @abstractmethod
    def set_action_space(self) -> None:
        ...

    def action_masks(self) -> list[bool]:
        ...

    def seed(self, seed: int = 1) -> list[int]:
        ...

    def tensorboard_log(self, metric: str, value: Any = ..., inc: bool = ..., category: str = ...) -> None:
        ...

    def reset_tensorboard_log(self) -> None:
        ...

    def reset(self, seed: int = ...) -> tuple[spaces.Box, dict]:
        ...

    @abstractmethod
    def step(self, action: int) -> tuple[spaces.Box, float, bool, dict]:
        ...

    def _get_observation(self) -> DataFrame:
        ...

    def get_trade_duration(self) -> int:
        ...

    def get_unrealized_profit(self) -> float:
        ...

    @abstractmethod
    def is_tradesignal(self, action: int) -> bool:
        ...

    def _is_valid(self, action: int) -> bool:
        ...

    def add_entry_fee(self, price: float) -> float:
        ...

    def add_exit_fee(self, price: float) -> float:
        ...

    def _update_history(self, info: dict) -> None:
        ...

    @abstractmethod
    def calculate_reward(self, action: int) -> float:
        ...

    def _update_unrealized_total_profit(self) -> None:
        ...

    def _update_total_profit(self) -> None:
        ...

    def current_price(self) -> float:
        ...

    def get_actions(self) -> type[BaseActions]:
        ...