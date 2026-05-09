from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import ANY, MagicMock, PropertyMock
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    TypeVar,
    Type,
    overload,
    Protocol,
    runtime_checkable,
)
from pytest import (
    fixture,
    FixtureRequest,
    LogCaptureFixture,
    MonkeyPatch,
    mark,
    raises,
)
from pandas import DataFrame
from filelock import Timeout
from skopt.space import Integer
from freqtrade.enums import ExitType, RunMode
from freqtrade.strategy import IntParameter
from freqtrade.util import dt_utc
from tests.conftest import (
    CURRENT_TEST_STRATEGY,
    EXMS,
    get_args,
    get_markets,
    log_has,
    log_has_re,
    patch_exchange,
    patched_configuration_load_config_file,
)

ResultMetrics = Dict[str, Union[float, int, timedelta, bool]]
Params = Dict[str, Union[float, int, str, None]]
Results = Dict[str, Union[float, str, ResultMetrics, Params]]

def generate_result_metrics() -> ResultMetrics:
    ...

def test_setup_hyperopt_configuration_without_arguments(
    mocker: MonkeyPatch,
    default_conf: dict,
    caplog: LogCaptureFixture,
) -> None:
    ...

def test_setup_hyperopt_configuration_with_arguments(
    mocker: MonkeyPatch,
    default_conf: dict,
    caplog: LogCaptureFixture,
) -> None:
    ...

def test_setup_hyperopt_configuration_stake_amount(
    mocker: MonkeyPatch,
    default_conf: dict,
) -> None:
    ...

def test_start_not_installed(
    mocker: MonkeyPatch,
    default_conf: dict,
    import_fails: fixture,
) -> None:
    ...

def test_start_no_hyperopt_allowed(
    mocker: MonkeyPatch,
    hyperopt_conf: dict,
    caplog: LogCaptureFixture,
) -> None:
    ...

def test_start_no_data(
    mocker: MonkeyPatch,
    hyperopt_conf: dict,
    tmp_path: Path,
) -> None:
    ...

def test_start_filelock(
    mocker: MonkeyPatch,
    hyperopt_conf: dict,
    caplog: LogCaptureFixture,
) -> None:
    ...

def test_log_results_if_loss_improves(
    hyperopt: fixture,
    capsys: fixture,
) -> None:
    ...

def test_no_log_if_loss_does_not_improve(
    hyperopt: fixture,
    caplog: LogCaptureFixture,
) -> None:
    ...

def test_roi_table_generation(
    hyperopt: fixture,
) -> None:
    ...

def test_params_no_optimize_details(
    hyperopt: fixture,
) -> None:
    ...

def test_start_calls_optimizer(
    mocker: MonkeyPatch,
    hyperopt_conf: dict,
    capsys: fixture,
) -> None:
    ...

def test_hyperopt_format_results(
    hyperopt: fixture,
) -> None:
    ...

def test_populate_indicators(
    hyperopt: fixture,
    testdatadir: Path,
) -> None:
    ...

def test_generate_optimizer(
    mocker: MonkeyPatch,
    hyperopt_conf: dict,
) -> None:
    ...

def test_clean_hyperopt(
    mocker: MonkeyPatch,
    hyperopt_conf: dict,
    caplog: LogCaptureFixture,
) -> None:
    ...

def test_print_json_spaces_all(
    mocker: MonkeyPatch,
    hyperopt_conf: dict,
    capsys: fixture,
) -> None:
    ...

def test_print_json_spaces_default(
    mocker: MonkeyPatch,
    hyperopt_conf: dict,
    capsys: fixture,
) -> None:
    ...

def test_print_json_spaces_roi_stoploss(
    mocker: MonkeyPatch,
    hyperopt_conf: dict,
    capsys: fixture,
) -> None:
    ...

def test_simplified_interface_roi_stoploss(
    mocker: MonkeyPatch,
    hyperopt_conf: dict,
    capsys: fixture,
) -> None:
    ...

def test_simplified_interface_all_failed(
    mocker: MonkeyPatch,
    hyperopt_conf: dict,
    caplog: LogCaptureFixture,
) -> None:
    ...

def test_simplified_interface_buy(
    mocker: MonkeyPatch,
    hyperopt_conf: dict,
    capsys: fixture,
) -> None:
    ...

def test_simplified_interface_sell(
    mocker: MonkeyPatch,
    hyperopt_conf: dict,
    capsys: fixture,
) -> None:
    ...

def test_simplified_interface_failed(
    mocker: MonkeyPatch,
    hyperopt_conf: dict,
    space: str,
) -> None:
    ...

def test_in_strategy_auto_hyperopt(
    mocker: MonkeyPatch,
    hyperopt_conf: dict,
    tmp_path: Path,
    fee: fixture,
) -> None:
    ...

def test_in_strategy_auto_hyperopt_with_parallel(
    mocker: MonkeyPatch,
    hyperopt_conf: dict,
    tmp_path: Path,
    fee: fixture,
) -> None:
    ...

def test_in_strategy_auto_hyperopt_per_epoch(
    mocker: MonkeyPatch,
    hyperopt_conf: dict,
    tmp_path: Path,
    fee: fixture,
) -> None:
    ...

def test_SKDecimal() -> None:
    ...

def test_stake_amount_unlimited_max_open_trades(
    mocker: MonkeyPatch,
    hyperopt_conf: dict,
    tmp_path: Path,
    fee: fixture,
) -> None:
    ...

def test_max_open_trades_dump(
    mocker: MonkeyPatch,
    hyperopt_conf: dict,
    tmp_path: Path,
    fee: fixture,
    capsys: fixture,
) -> None:
    ...

def test_max_open_trades_consistency(
    mocker: MonkeyPatch,
    hyperopt_conf: dict,
    tmp_path: Path,
    fee: fixture,
) -> None:
    ...