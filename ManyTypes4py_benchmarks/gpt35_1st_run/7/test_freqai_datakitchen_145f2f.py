from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import pytest

def test_create_fulltimerange(timerange: str, train_period_days: int, expected_result: str, freqai_conf: dict, mocker: MagicMock, caplog: MagicMock) -> None:
def test_create_fulltimerange_incorrect_backtest_period(mocker: MagicMock, freqai_conf: dict) -> None:
def test_split_timerange(mocker: MagicMock, freqai_conf: dict, timerange: str, train_period_days: int, backtest_period_days: float, expected_result: int) -> None:
def test_check_if_model_expired(mocker: MagicMock, freqai_conf: dict) -> None:
def test_filter_features(mocker: MagicMock, freqai_conf: dict) -> None:
def test_make_train_test_datasets(mocker: MagicMock, freqai_conf: dict) -> None:
def test_get_full_model_path(mocker: MagicMock, freqai_conf: dict, model: str) -> None:
def test_get_pair_data_for_features_with_prealoaded_data(mocker: MagicMock, freqai_conf: dict) -> None:
def test_get_pair_data_for_features_without_preloaded_data(mocker: MagicMock, freqai_conf: dict) -> None:
def test_populate_features(mocker: MagicMock, freqai_conf: dict) -> None:
