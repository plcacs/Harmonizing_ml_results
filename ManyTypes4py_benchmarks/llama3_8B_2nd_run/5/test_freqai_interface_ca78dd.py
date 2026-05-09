from typing import Dict, Any, List
import logging
import shutil
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from freqtrade.configuration import TimeRange
from freqtrade.data.dataprovider import DataProvider
from freqtrade.enums import RunMode
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.utils import download_all_data_for_training, get_required_data_timerange
from freqtrade.optimize.backtesting import Backtesting
from freqtrade.persistence import Trade
from freqtrade.plugins.pairlistmanager import PairListManager
from tests.conftest import EXMS, create_mock_trades, get_patched_exchange, is_arm, is_mac, log_has_re
from tests.freqai.conftest import get_patched_freqai_strategy, make_rl_config, mock_pytorch_mlp_model_training_parameters

def can_run_model(model: str) -> None:
    # ...

@pytest.mark.parametrize('model, pca, dbscan, float32, can_short, shuffle, buffer, noise', [
    # ...
], ids=['test1', 'test2', ...])
def test_extract_data_and_train_model_Standard(mocker: Any, freqai_conf: Dict[str, Any], model: str, pca: bool, dbscan: bool, float32: bool, can_short: bool, shuffle: bool, buffer: int, noise: float) -> None:
    # ...

@pytest.mark.parametrize('model, strat', [
    # ...
], ids=['test1', 'test2', ...])
def test_extract_data_and_train_model_MultiTargets(mocker: Any, freqai_conf: Dict[str, Any], model: str, strat: str) -> None:
    # ...

@pytest.mark.parametrize('model', [
    # ...
], ids=['test1', 'test2', ...])
def test_extract_data_and_train_model_Classifiers(mocker: Any, freqai_conf: Dict[str, Any], model: str) -> None:
    # ...

@pytest.mark.parametrize('model, num_files, strat', [
    # ...
], ids=['test1', 'test2', ...])
def test_start_backtesting(mocker: Any, freqai_conf: Dict[str, Any], model: str, num_files: int, strat: str) -> None:
    # ...

def test_start_backtesting_subdaily_backtest_period(mocker: Any, freqai_conf: Dict[str, Any]) -> None:
    # ...

def test_backtesting_fit_live_predictions(mocker: Any, freqai_conf: Dict[str, Any], caplog: logging.LoggerAdapter) -> None:
    # ...

def test_plot_feature_importance(mocker: Any, freqai_conf: Dict[str, Any]) -> None:
    # ...

@pytest.mark.parametrize('timeframes, corr_pairs', [
    # ...
], ids=['test1', 'test2', ...])
def test_freqai_informative_pairs(mocker: Any, freqai_conf: Dict[str, Any], timeframes: List[str], corr_pairs: List[str]) -> None:
    # ...

def test_start_set_train_queue(mocker: Any, freqai_conf: Dict[str, Any], caplog: logging.LoggerAdapter) -> None:
    # ...

def test_get_required_data_timerange(mocker: Any, freqai_conf: Dict[str, Any]) -> TimeRange:
    # ...

def test_download_all_data_for_training(mocker: Any, freqai_conf: Dict[str, Any], caplog: logging.LoggerAdapter, tmp_path: Path) -> None:
    # ...

@pytest.mark.usefixtures('init_persistence')
@pytest.mark.parametrize('dp_exists', [False, True])
def test_get_state_info(mocker: Any, freqai_conf: Dict[str, Any], dp_exists: bool, caplog: logging.LoggerAdapter, tickers: Dict[str, Any]) -> None:
    # ...
