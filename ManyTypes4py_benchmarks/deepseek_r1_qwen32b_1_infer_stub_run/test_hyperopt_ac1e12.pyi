from datetime import datetime, timedelta
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
import pytest
from filelock import Timeout
from functools import wraps
from unittest.mock import ANY, MagicMock, PropertyMock
import pandas as pd
from freqtrade.enums import ExitType, RunMode
from freqtrade.exceptions import OperationalException
from freqtrade.optimize.hyperopt import Hyperopt
from freqtrade.optimize.hyperopt.hyperopt_auto import HyperOptAuto
from freqtrade.strategy import IntParameter
from tests.conftest import EXMS, get_args, get_markets, log_has, log_has_re, patch_exchange, patched_configuration_load_config_file

def generate_result_metrics() -> Dict[str, Union[int, float, timedelta, bool]]:
    ...

def test_setup_hyperopt_configuration_without_arguments(
    mocker: pytest.fixture,
    default_conf: Dict[str, Any],
    caplog: pytest.fixture,
) -> None:
    ...

def test_setup_hyperopt_configuration_with_arguments(
    mocker: pytest.fixture,
    default_conf: Dict[str, Any],
    caplog: pytest.fixture,
) -> None:
    ...

def test_setup_hyperopt_configuration_stake_amount(
    mocker: pytest.fixture,
    default_conf: Dict[str, Any],
) -> None:
    ...

def test_start_not_installed(
    mocker: pytest.fixture,
    default_conf: Dict[str, Any],
    import_fails: pytest.fixture,
) -> None:
    ...

def test_start_no_hyperopt_allowed(
    mocker: pytest.fixture,
    hyperopt_conf: Dict[str, Any],
    caplog: pytest.fixture,
) -> None:
    ...

def test_start_no_data(
    mocker: pytest.fixture,
    hyperopt_conf: Dict[str, Any],
    tmp_path: Path,
) -> None:
    ...

def test_start_filelock(
    mocker: pytest.fixture,
    hyperopt_conf: Dict[str, Any],
    caplog: pytest.fixture,
) -> None:
    ...

def test_log_results_if_loss_improves(
    hyperopt: Hyperopt,
    capsys: pytest.fixture,
) -> None:
    ...

def test_no_log_if_loss_does_not_improve(
    hyperopt: Hyperopt,
    caplog: pytest.fixture,
) -> None:
    ...

def test_roi_table_generation(
    hyperopt: Hyperopt,
) -> Dict[int, float]:
    ...

def test_params_no_optimize_details(
    hyperopt: Hyperopt,
) -> Dict[str, Dict[str, Union[int, float, bool]]]:
    ...

def test_start_calls_optimizer(
    mocker: pytest.fixture,
    hyperopt_conf: Dict[str, Any],
    capsys: pytest.fixture,
) -> None:
    ...

def test_hyperopt_format_results(
    hyperopt: Hyperopt,
) -> None:
    ...

def test_populate_indicators(
    hyperopt: Hyperopt,
    testdatadir: Path,
) -> None:
    ...

def test_generate_optimizer(
    mocker: pytest.fixture,
    hyperopt_conf: Dict[str, Any],
) -> None:
    ...

def test_clean_hyperopt(
    mocker: pytest.fixture,
    hyperopt_conf: Dict[str, Any],
    caplog: pytest.fixture,
) -> None:
    ...

def test_print_json_spaces_all(
    mocker: pytest.fixture,
    hyperopt_conf: Dict[str, Any],
    capsys: pytest.fixture,
) -> None:
    ...

def test_print_json_spaces_default(
    mocker: pytest.fixture,
    hyperopt_conf: Dict[str, Any],
    capsys: pytest.fixture,
) -> None:
    ...

def test_print_json_spaces_roi_stoploss(
    mocker: pytest.fixture,
    hyperopt_conf: Dict[str, Any],
    capsys: pytest.fixture,
) -> None:
    ...

def test_simplified_interface_roi_stoploss(
    mocker: pytest.fixture,
    hyperopt_conf: Dict[str, Any],
    capsys: pytest.fixture,
) -> None:
    ...

def test_simplified_interface_all_failed(
    mocker: pytest.fixture,
    hyperopt_conf: Dict[str, Any],
    caplog: pytest.fixture,
) -> None:
    ...

def test_simplified_interface_buy(
    mocker: pytest.fixture,
    hyperopt_conf: Dict[str, Any],
    capsys: pytest.fixture,
) -> None:
    ...

def test_simplified_interface_sell(
    mocker: pytest.fixture,
    hyperopt_conf: Dict[str, Any],
    capsys: pytest.fixture,
) -> None:
    ...

@pytest.mark.parametrize('space', ['buy', 'sell', 'protection'])
def test_simplified_interface_failed(
    mocker: pytest.fixture,
    hyperopt_conf: Dict[str, Any],
    space: str,
) -> None:
    ...

def test_in_strategy_auto_hyperopt(
    mocker: pytest.fixture,
    hyperopt_conf: Dict[str, Any],
    tmp_path: Path,
    fee: Callable,
) -> None:
    ...

@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_in_strategy_auto_hyperopt_with_parallel(
    mocker: pytest.fixture,
    hyperopt_conf: Dict[str, Any],
    tmp_path: Path,
    fee: Callable,
) -> None:
    ...

def test_in_strategy_auto_hyperopt_per_epoch(
    mocker: pytest.fixture,
    hyperopt_conf: Dict[str, Any],
    tmp_path: Path,
    fee: Callable,
) -> None:
    ...

def test_SKDecimal() -> None:
    ...

def test_stake_amount_unlimited_max_open_trades(
    mocker: pytest.fixture,
    hyperopt_conf: Dict[str, Any],
    tmp_path: Path,
    fee: Callable,
) -> None:
    ...

def test_max_open_trades_dump(
    mocker: pytest.fixture,
    hyperopt_conf: Dict[str, Any],
    tmp_path: Path,
    fee: Callable,
    capsys: pytest.fixture,
) -> None:
    ...

def test_max_open_trades_consistency(
    mocker: pytest.fixture,
    hyperopt_conf: Dict[str, Any],
    tmp_path: Path,
    fee: Callable,
) -> None:
    ...