```python
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import ANY, MagicMock
import pandas as pd
import pytest
from filelock import Timeout
from freqtrade.enums import ExitType, RunMode
from freqtrade.exceptions import OperationalException

def generate_result_metrics() -> dict[str, Any]: ...

def test_setup_hyperopt_configuration_without_arguments(
    mocker: Any,
    default_conf: dict[str, Any],
    caplog: Any
) -> None: ...

def test_setup_hyperopt_configuration_with_arguments(
    mocker: Any,
    default_conf: dict[str, Any],
    caplog: Any
) -> None: ...

def test_setup_hyperopt_configuration_stake_amount(
    mocker: Any,
    default_conf: dict[str, Any]
) -> None: ...

def test_start_not_installed(
    mocker: Any,
    default_conf: dict[str, Any],
    import_fails: Any
) -> None: ...

def test_start_no_hyperopt_allowed(
    mocker: Any,
    hyperopt_conf: dict[str, Any],
    caplog: Any
) -> None: ...

def test_start_no_data(
    mocker: Any,
    hyperopt_conf: dict[str, Any],
    tmp_path: Path
) -> None: ...

def test_start_filelock(
    mocker: Any,
    hyperopt_conf: dict[str, Any],
    caplog: Any
) -> None: ...

def test_log_results_if_loss_improves(
    hyperopt: Any,
    capsys: Any
) -> None: ...

def test_no_log_if_loss_does_not_improve(
    hyperopt: Any,
    caplog: Any
) -> None: ...

def test_roi_table_generation(hyperopt: Any) -> None: ...

def test_params_no_optimize_details(hyperopt: Any) -> None: ...

def test_start_calls_optimizer(
    mocker: Any,
    hyperopt_conf: dict[str, Any],
    capsys: Any
) -> None: ...

def test_hyperopt_format_results(hyperopt: Any) -> None: ...

def test_populate_indicators(
    hyperopt: Any,
    testdatadir: Path
) -> None: ...

def test_generate_optimizer(
    mocker: Any,
    hyperopt_conf: dict[str, Any]
) -> None: ...

def test_clean_hyperopt(
    mocker: Any,
    hyperopt_conf: dict[str, Any],
    caplog: Any
) -> None: ...

def test_print_json_spaces_all(
    mocker: Any,
    hyperopt_conf: dict[str, Any],
    capsys: Any
) -> None: ...

def test_print_json_spaces_default(
    mocker: Any,
    hyperopt_conf: dict[str, Any],
    capsys: Any
) -> None: ...

def test_print_json_spaces_roi_stoploss(
    mocker: Any,
    hyperopt_conf: dict[str, Any],
    capsys: Any
) -> None: ...

def test_simplified_interface_roi_stoploss(
    mocker: Any,
    hyperopt_conf: dict[str, Any],
    capsys: Any
) -> None: ...

def test_simplified_interface_all_failed(
    mocker: Any,
    hyperopt_conf: dict[str, Any],
    caplog: Any
) -> None: ...

def test_simplified_interface_buy(
    mocker: Any,
    hyperopt_conf: dict[str, Any],
    capsys: Any
) -> None: ...

def test_simplified_interface_sell(
    mocker: Any,
    hyperopt_conf: dict[str, Any],
    capsys: Any
) -> None: ...

def test_simplified_interface_failed(
    mocker: Any,
    hyperopt_conf: dict[str, Any],
    space: str
) -> None: ...

def test_in_strategy_auto_hyperopt(
    mocker: Any,
    hyperopt_conf: dict[str, Any],
    tmp_path: Path,
    fee: Any
) -> None: ...

def test_in_strategy_auto_hyperopt_with_parallel(
    mocker: Any,
    hyperopt_conf: dict[str, Any],
    tmp_path: Path,
    fee: Any
) -> None: ...

def test_in_strategy_auto_hyperopt_per_epoch(
    mocker: Any,
    hyperopt_conf: dict[str, Any],
    tmp_path: Path,
    fee: Any
) -> None: ...

def test_SKDecimal() -> None: ...

def test_stake_amount_unlimited_max_open_trades(
    mocker: Any,
    hyperopt_conf: dict[str, Any],
    tmp_path: Path,
    fee: Any
) -> None: ...

def test_max_open_trades_dump(
    mocker: Any,
    hyperopt_conf: dict[str, Any],
    tmp_path: Path,
    fee: Any,
    capsys: Any
) -> None: ...

def test_max_open_trades_consistency(
    mocker: Any,
    hyperopt_conf: dict[str, Any],
    tmp_path: Path,
    fee: Any
) -> None: ...
```