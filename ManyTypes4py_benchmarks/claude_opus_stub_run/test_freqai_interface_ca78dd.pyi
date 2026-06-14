import logging
from pathlib import Path
from typing import Any

import pytest


def can_run_model(model: str) -> None: ...

def test_extract_data_and_train_model_Standard(
    mocker: Any,
    freqai_conf: dict[str, Any],
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
    mocker: Any,
    freqai_conf: dict[str, Any],
    model: str,
    strat: str,
) -> None: ...

def test_extract_data_and_train_model_Classifiers(
    mocker: Any,
    freqai_conf: dict[str, Any],
    model: str,
) -> None: ...

def test_start_backtesting(
    mocker: Any,
    freqai_conf: dict[str, Any],
    model: str,
    num_files: int,
    strat: str,
    caplog: pytest.LogCaptureFixture,
) -> None: ...

def test_start_backtesting_subdaily_backtest_period(
    mocker: Any,
    freqai_conf: dict[str, Any],
) -> None: ...

def test_start_backtesting_from_existing_folder(
    mocker: Any,
    freqai_conf: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
) -> None: ...

def test_backtesting_fit_live_predictions(
    mocker: Any,
    freqai_conf: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
) -> None: ...

def test_plot_feature_importance(
    mocker: Any,
    freqai_conf: dict[str, Any],
) -> None: ...

def test_freqai_informative_pairs(
    mocker: Any,
    freqai_conf: dict[str, Any],
    timeframes: list[str],
    corr_pairs: list[str],
) -> None: ...

def test_start_set_train_queue(
    mocker: Any,
    freqai_conf: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
) -> None: ...

def test_get_required_data_timerange(
    mocker: Any,
    freqai_conf: dict[str, Any],
) -> None: ...

def test_download_all_data_for_training(
    mocker: Any,
    freqai_conf: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None: ...

def test_get_state_info(
    mocker: Any,
    freqai_conf: dict[str, Any],
    dp_exists: bool,
    caplog: pytest.LogCaptureFixture,
    tickers: Any,
) -> None: ...