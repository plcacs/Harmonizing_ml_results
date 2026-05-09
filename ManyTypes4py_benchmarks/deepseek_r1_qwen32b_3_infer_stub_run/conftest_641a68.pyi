import argparse
import json
import logging
import platform
import re
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, Mock, PropertyMock
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pytest
from xdist.scheduler.loadscope import LoadScopeScheduling
from freqtrade import constants
from freqtrade.commands import Arguments
from freqtrade.data.converter import ohlcv_to_dataframe, trades_list_to_df
from freqtrade.edge import PairInfo
from freqtrade.enums import CandleType, MarginMode, RunMode, SignalDirection, TradingMode
from freqtrade.exchange import Exchange, timeframe_to_minutes, timeframe_to_seconds
from freqtrade.freqtradebot import FreqtradeBot
from freqtrade.persistence import LocalTrade, Order, Trade, init_db
from freqtrade.resolvers import ExchangeResolver
from freqtrade.util import dt_now, dt_ts
from freqtrade.worker import Worker

class FixtureScheduler(LoadScopeScheduling):
    def _split_scope(self, nodeid: str) -> str: ...

def pytest_addoption(parser: argparse.ArgumentParser) -> None: ...

def pytest_configure(config: pytest.Config) -> None: ...

def pytest_xdist_make_scheduler(config: pytest.Config, log: Any) -> FixtureScheduler: ...

def log_has(line: str, logs: pytest.LogCaptureFixture) -> bool: ...

def log_has_when(line: str, logs: pytest.LogCaptureFixture, when: str) -> bool: ...