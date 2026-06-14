from datetime import timedelta
from typing import Any, Dict

from _pytest.capture import CaptureFixture
from _pytest.logging import LogCaptureFixture
from pathlib import Path
from pytest_mock import MockerFixture

from freqtrade.optimize.hyperopt import Hyperopt


def generate_result_metrics() -> Dict[str, Any]: ...

def test_setup_hyperopt_configuration_without_arguments(
    mocker: MockerFixture, default_conf: Dict[str, Any], caplog: LogCaptureFixture
) -> None: ...

def test_setup_hyperopt_configuration_with_arguments(
    mocker: MockerFixture, default_conf: Dict[str, Any], caplog: LogCaptureFixture
) -> None: ...

def test_setup_hyperopt_configuration_stake_amount(
    mocker: MockerFixture, default_conf: Dict[str, Any]
) -> None: ...

def test_start_not_installed(
    mocker: MockerFixture, default_conf: Dict[str, Any], import_fails: Any
) -> None: ...

def test_start_no_hyperopt_allowed(
    mocker: MockerFixture, hyperopt_conf: Dict[str, Any], caplog: LogCaptureFixture
) -> None: ...

def test_start_no_data(
    mocker: MockerFixture, hyperopt_conf: Dict[str, Any], tmp_path: Path
) -> None: ...

def test_start_filelock(
    mocker: MockerFixture, hyperopt_conf: Dict[str, Any], caplog: LogCaptureFixture
) -> None: ...

def test_log_results_if_loss_improves(
    hyperopt: Hyperopt, capsys: CaptureFixture[str]
) -> None: ...

def test_no_log_if_loss_does_not_improve(
    hyperopt: Hyperopt, caplog: LogCaptureFixture
) -> None: ...

def test_roi_table_generation(hyperopt: Hyperopt) -> None: ...

def test_params_no_optimize_details(hyperopt: Hyperopt) -> None: ...

def test_start_calls_optimizer(
    mocker: MockerFixture, hyperopt_conf: Dict[str, Any], capsys: CaptureFixture[str]
) -> None: ...

def test_hyperopt_format_results(hyperopt: Hyperopt) -> None: ...

def test_populate_indicators(
    hyperopt: Hyperopt, testdatadir: Path
) -> None: ...

def test_generate_optimizer(
    mocker: MockerFixture, hyperopt_conf: Dict[str, Any]
) -> None: ...

def test_clean_hyperopt(
    mocker: MockerFixture, hyperopt_conf: Dict[str, Any], caplog: LogCaptureFixture
) -> None: ...

def test_print_json_spaces_all(
    mocker: MockerFixture, hyperopt_conf: Dict[str, Any], capsys: CaptureFixture[str]
) -> None: ...

def test_print_json_spaces_default(
    mocker: MockerFixture, hyperopt_conf: Dict[str, Any], capsys: CaptureFixture[str]
) -> None: ...

def test_print_json_spaces_roi_stoploss(
    mocker: MockerFixture, hyperopt_conf: Dict[str, Any], capsys: CaptureFixture[str]
) -> None: ...

def test_simplified_interface_roi_stoploss(
    mocker: MockerFixture, hyperopt_conf: Dict[str, Any], capsys: CaptureFixture[str]
) -> None: ...

def test_simplified_interface_all_failed(
    mocker: MockerFixture, hyperopt_conf: Dict[str, Any], caplog: LogCaptureFixture
) -> None: ...

def test_simplified_interface_buy(
    mocker: MockerFixture, hyperopt_conf: Dict[str, Any], capsys: CaptureFixture[str]
) -> None: ...

def test_simplified_interface_sell(
    mocker: MockerFixture, hyperopt_conf: Dict[str, Any], capsys: CaptureFixture[str]
) -> None: ...

def test_simplified_interface_failed(
    mocker: MockerFixture, hyperopt_conf: Dict[str, Any], space: str
) -> None: ...

def test_in_strategy_auto_hyperopt(
    mocker: MockerFixture, hyperopt_conf: Dict[str, Any], tmp_path: Path, fee: Any
) -> None: ...

def test_in_strategy_auto_hyperopt_with_parallel(
    mocker: MockerFixture, hyperopt_conf: Dict[str, Any], tmp_path: Path, fee: Any
) -> None: ...

def test_in_strategy_auto_hyperopt_per_epoch(
    mocker: MockerFixture, hyperopt_conf: Dict[str, Any], tmp_path: Path, fee: Any
) -> None: ...

def test_SKDecimal() -> None: ...

def test_stake_amount_unlimited_max_open_trades(
    mocker: MockerFixture, hyperopt_conf: Dict[str, Any], tmp_path: Path, fee: Any
) -> None: ...

def test_max_open_trades_dump(
    mocker: MockerFixture,
    hyperopt_conf: Dict[str, Any],
    tmp_path: Path,
    fee: Any,
    capsys: CaptureFixture[str],
) -> None: ...

def test_max_open_trades_consistency(
    mocker: MockerFixture, hyperopt_conf: Dict[str, Any], tmp_path: Path, fee: Any
) -> None: ...