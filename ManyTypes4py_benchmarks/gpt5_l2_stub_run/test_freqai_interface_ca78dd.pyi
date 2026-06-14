from pathlib import Path
from typing import Any, Callable, Dict, List
from _pytest.logging import LogCaptureFixture
from pytest_mock import MockerFixture

def can_run_model(model: str) -> None: ...
def test_extract_data_and_train_model_Standard(
    mocker: MockerFixture,
    freqai_conf: Dict[str, Any],
    model: str,
    pca: bool,
    dbscan: bool,
    float32: bool,
    can_short: bool,
    shuffle: bool,
    buffer: int,
    noise: float,
) -> None: ...
def test_extract_data_and_train_model_MultiTargets(
    mocker: MockerFixture,
    freqai_conf: Dict[str, Any],
    model: str,
    strat: str,
) -> None: ...
def test_extract_data_and_train_model_Classifiers(
    mocker: MockerFixture,
    freqai_conf: Dict[str, Any],
    model: str,
) -> None: ...
def test_start_backtesting(
    mocker: MockerFixture,
    freqai_conf: Dict[str, Any],
    model: str,
    num_files: int,
    strat: str,
    caplog: LogCaptureFixture,
) -> None: ...
def test_start_backtesting_subdaily_backtest_period(
    mocker: MockerFixture,
    freqai_conf: Dict[str, Any],
) -> None: ...
def test_start_backtesting_from_existing_folder(
    mocker: MockerFixture,
    freqai_conf: Dict[str, Any],
    caplog: LogCaptureFixture,
) -> None: ...
def test_backtesting_fit_live_predictions(
    mocker: MockerFixture,
    freqai_conf: Dict[str, Any],
    caplog: LogCaptureFixture,
) -> None: ...
def test_plot_feature_importance(
    mocker: MockerFixture,
    freqai_conf: Dict[str, Any],
) -> None: ...
def test_freqai_informative_pairs(
    mocker: MockerFixture,
    freqai_conf: Dict[str, Any],
    timeframes: List[str],
    corr_pairs: List[str],
) -> None: ...
def test_start_set_train_queue(
    mocker: MockerFixture,
    freqai_conf: Dict[str, Any],
    caplog: LogCaptureFixture,
) -> None: ...
def test_get_required_data_timerange(
    mocker: MockerFixture,
    freqai_conf: Dict[str, Any],
) -> None: ...
def test_download_all_data_for_training(
    mocker: MockerFixture,
    freqai_conf: Dict[str, Any],
    caplog: LogCaptureFixture,
    tmp_path: Path,
) -> None: ...
def test_get_state_info(
    mocker: MockerFixture,
    freqai_conf: Dict[str, Any],
    dp_exists: bool,
    caplog: LogCaptureFixture,
    tickers: Callable[[], Dict[str, Any]],
) -> None: ...