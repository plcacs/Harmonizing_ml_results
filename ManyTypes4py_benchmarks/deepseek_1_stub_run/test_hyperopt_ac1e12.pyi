```python
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import MagicMock
import pandas as pd
import pytest
from filelock import Timeout
from skopt.space import Integer
from freqtrade.commands.optimize_commands import setup_optimize_configuration, start_hyperopt
from freqtrade.data.history import load_data
from freqtrade.enums import ExitType, RunMode
from freqtrade.exceptions import OperationalException
from freqtrade.optimize.hyperopt import Hyperopt
from freqtrade.optimize.hyperopt.hyperopt_auto import HyperOptAuto
from freqtrade.optimize.hyperopt_tools import HyperoptTools
from freqtrade.optimize.optimize_reports import generate_strategy_stats
from freqtrade.optimize.space import SKDecimal
from freqtrade.strategy import IntParameter
from freqtrade.util import dt_utc
from tests.conftest import CURRENT_TEST_STRATEGY, EXMS, get_args, get_markets, log_has, log_has_re, patch_exchange, patched_configuration_load_config_file

def generate_result_metrics() -> Dict[str, Any]: ...

def test_setup_hyperopt_configuration_without_arguments(
    mocker: Any,
    default_conf: Dict[str, Any],
    caplog: Any
) -> None: ...

def test_setup_hyperopt_configuration_with_arguments(
    mocker: Any,
    default_conf: Dict[str, Any],
    caplog: Any
) -> None: ...

def test_setup_hyperopt_configuration_stake_amount(
    mocker: Any,
    default_conf: Dict[str, Any]
) -> None: ...

def test_start_not_installed(
    mocker: Any,
    default_conf: Dict[str, Any],
    import_fails: Any
) -> None: ...

def test_start_no_hyperopt_allowed(
    mocker: Any,
    hyperopt_conf: Dict[str, Any],
    caplog: Any
) -> None: ...

def test_start_no_data(
    mocker: Any,
    hyperopt_conf: Dict[str, Any],
    tmp_path: Path
) -> None: ...

def test_start_filelock(
    mocker: Any,
    hyperopt_conf: Dict[str, Any],
    caplog: Any
) -> None: ...

def test_log_results_if_loss_improves(
    hyperopt: Hyperopt,
    capsys: Any
) -> None: ...

def test_no_log_if_loss_does_not_improve(
    hyperopt: Hyperopt,
    caplog: Any
) -> None: ...

def test_roi_table_generation(
    hyperopt: Hyperopt
) -> None: ...

def test_params_no_optimize_details(
    hyperopt: Hyperopt
) -> None: ...

def test_start_calls_optimizer(
    mocker: Any,
    hyperopt_conf: Dict[str, Any],
    capsys: Any
) -> None: ...

def test_hyperopt_format_results(
    hyperopt: Hyperopt
) -> None: ...

def test_populate_indicators(
    hyperopt: Hyperopt,
    testdatadir: Path
) -> None: ...

def test_generate_optimizer(
    mocker: Any,
    hyperopt_conf: Dict[str, Any]
) -> None: ...

def test_clean_hyperopt(
    mocker: Any,
    hyperopt_conf: Dict[str, Any],
    caplog: Any
) -> None: ...

def test_print_json_spaces_all(
    mocker: Any,
    hyperopt_conf: Dict[str, Any],
    capsys: Any
) -> None: ...

def test_print_json_spaces_default(
    mocker: Any,
    hyperopt_conf: Dict[str, Any],
    capsys: Any
) -> None: ...

def test_print_json_spaces_roi_stoploss(
    mocker: Any,
    hyperopt_conf: Dict[str, Any],
    capsys: Any
) -> None: ...

def test_simplified_interface_roi_stoploss(
    mocker: Any,
    hyperopt_conf: Dict[str, Any],
    capsys: Any
) -> None: ...

def test_simplified_interface_all_failed(
    mocker: Any,
    hyperopt_conf: Dict[str, Any],
    caplog: Any
) -> None: ...

def test_simplified_interface_buy(
    mocker: Any,
    hyperopt_conf: Dict[str, Any],
    capsys: Any
) -> None: ...

def test_simplified_interface_sell(
    mocker: Any,
    hyperopt_conf: Dict[str, Any],
    capsys: Any
) -> None: ...

@pytest.mark.parametrize('space', ['buy', 'sell', 'protection'])
def test_simplified_interface_failed(
    mocker: Any,
    hyperopt_conf: Dict[str, Any],
    space: str
) -> None: ...

def test_in_strategy_auto_hyperopt(
    mocker: Any,
    hyperopt_conf: Dict[str, Any],
    tmp_path: Path,
    fee: Any
) -> None: ...

@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_in_strategy_auto_hyperopt_with_parallel(
    mocker: Any,
    hyperopt_conf: Dict[str, Any],
    tmp_path: Path,
    fee: Any
) -> None: ...

def test_in_strategy_auto_hyperopt_per_epoch(
    mocker: Any,
    hyperopt_conf: Dict[str, Any],
    tmp_path: Path,
    fee: Any
) -> None: ...

def test_SKDecimal() -> None: ...

def test_stake_amount_unlimited_max_open_trades(
    mocker: Any,
    hyperopt_conf: Dict[str, Any],
    tmp_path: Path,
    fee: Any
) -> None: ...

def test_max_open_trades_dump(
    mocker: Any,
    hyperopt_conf: Dict[str, Any],
    tmp_path: Path,
    fee: Any,
    capsys: Any
) -> None: ...

def test_max_open_trades_consistency(
    mocker: Any,
    hyperopt_conf: Dict[str, Any],
    tmp_path: Path,
    fee: Any
) -> None: ...
```