import numpy as np
import pandas as pd
import pytest
from freqtrade.data.dataprovider import DataProvider
from freqtrade.enums import CandleType
from freqtrade.resolvers.strategy_resolver import StrategyResolver
from freqtrade.strategy import merge_informative_pair, stoploss_from_absolute, stoploss_from_open
from tests.conftest import generate_test_data, get_patched_exchange

def test_merge_informative_pair() -> None:
    # ... (rest of the function remains the same)

def test_merge_informative_pair_weekly() -> None:
    # ... (rest of the function remains the same)

def test_merge_informative_pair_monthly() -> None:
    # ... (rest of the function remains the same)

def test_merge_informative_pair_same() -> None:
    # ... (rest of the function remains the same)

def test_merge_informative_pair_lower() -> None:
    # ... (rest of the function remains the same)

def test_merge_informative_pair_empty() -> None:
    # ... (rest of the function remains the same)

def test_merge_informative_pair_suffix() -> None:
    # ... (rest of the function remains the same)

def test_stoploss_from_open(side: str, rel_stop: float, curr_profit: float, leverage: float, expected: float) -> None:
    # ... (rest of the function remains the same)

def test_stoploss_from_absolute() -> None:
    # ... (rest of the function remains the same)

def test_informative_decorator(mocker, default_conf_usdt: dict, trading_mode: str) -> None:
    # ... (rest of the function remains the same)
