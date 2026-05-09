import logging
import shutil
from pathlib import Path
from unittest.mock import MagicMock
from typing import Any, List, Optional, Union
import pytest
from freqtrade.configuration import TimeRange
from freqtrade.data.dataprovider import DataProvider
from freqtrade.enums import RunMode
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.optimize.backtesting import Backtesting
from freqtrade.persistence import Trade
from freqtrade.plugins.pairlistmanager import PairListManager

def can_run_model(model: str) -> None: ...

@pytest.mark.parametrize('model, pca, dbscan, float32, can_short, shuffle, buffer, noise', [('LightGBMRegressor', True, False, True, True, False, 0, 0), ('XGBoostRegressor', False, True, False, True, False, 10, 0.05), ('XGBoostRFRegressor', False, False, False, True, False, 0, 0), ('CatboostRegressor', False, False, False, True, True, 0, 0), ('PyTorchMLPRegressor', False, False, False, False, False, 0, 0), ('PyTorchTransformerRegressor', False, False, False, False, False, 0, 0), ('ReinforcementLearner', False, True, False, True, False, 0, 0), ('ReinforcementLearner_multiproc', False, False, False, True, False, 0, 0), ('ReinforcementLearner_test_3ac', False, False, False, False, False, 0, 0), ('ReinforcementLearner_test_3ac', False, False, False, True, False, 0, 0), ('ReinforcementLearner_test_4ac', False, False, False, True, False, 0, 0)])
def test_extract_data_and_train_model_Standard(mocker: pytest.MonkeyPatch, freqai_conf: dict, model: str, pca: bool, dbscan: bool, float32: bool, can_short: bool, shuffle: bool, buffer: int, noise: float) -> None: ...

@pytest.mark.parametrize('model, strat', [('LightGBMRegressorMultiTarget', 'freqai_test_multimodel_strat'), ('XGBoostRegressorMultiTarget', 'freqai_test_multimodel_strat'), ('CatboostRegressorMultiTarget', 'freqai_test_multimodel_strat'), ('LightGBMClassifierMultiTarget', 'freqai_test_multimodel_classifier_strat'), ('CatboostClassifierMultiTarget', 'freqai_test_multimodel_classifier_strat')])
def test_extract_data_and_train_model_MultiTargets(mocker: pytest.MonkeyPatch, freqai_conf: dict, model: str, strat: str) -> None: ...

@pytest.mark.parametrize('model', ['LightGBMClassifier', 'CatboostClassifier', 'XGBoostClassifier', 'XGBoostRFClassifier', 'SKLearnRandomForestClassifier', 'PyTorchMLPClassifier'])
def test_extract_data_and_train_model_Classifiers(mocker: pytest.MonkeyPatch, freqai_conf: dict, model: str) -> None: ...

@pytest.mark.parametrize('model, num_files, strat', [('LightGBMRegressor', 2, 'freqai_test_strat'), ('XGBoostRegressor', 2, 'freqai_test_strat'), ('CatboostRegressor', 2, 'freqai_test_strat'), ('PyTorchMLPRegressor', 2, 'freqai_test_strat'), ('PyTorchTransformerRegressor', 2, 'freqai_test_strat'), ('ReinforcementLearner', 3, 'freqai_rl_test_strat'), ('XGBoostClassifier', 2, 'freqai_test_classifier'), ('LightGBMClassifier', 2, 'freqai_test_classifier'), ('CatboostClassifier', 2, 'freqai_test_classifier'), ('PyTorchMLPClassifier', 2, 'freqai_test_classifier')])
def test_start_backtesting(mocker: pytest.MonkeyPatch, freqai_conf: dict, model: str, num_files: int, strat: str, caplog: pytest.LogCaptureFixture) -> None: ...

def test_start_backtesting_subdaily_backtest_period(mocker: pytest.MonkeyPatch, freqai_conf: dict) -> None: ...

def test_start_backtesting_from_existing_folder(mocker: pytest.MonkeyPatch, freqai_conf: dict, caplog: pytest.LogCaptureFixture) -> None: ...

def test_backtesting_fit_live_predictions(mocker: pytest.MonkeyPatch, freqai_conf: dict, caplog: pytest.LogCaptureFixture) -> None: ...

def test_plot_feature_importance(mocker: pytest.MonkeyPatch, freqai_conf: dict) -> None: ...

@pytest.mark.parametrize('timeframes,corr_pairs', [(['5m'], ['ADA/BTC', 'DASH/BTC']), (['5m'], ['ADA/BTC', 'DASH/BTC', 'ETH/USDT']), (['5m', '15m'], ['ADA/BTC', 'DASH/BTC', 'ETH/USDT'])])
def test_freqai_informative_pairs(mocker: pytest.MonkeyPatch, freqai_conf: dict, timeframes: List[str], corr_pairs: List[str]) -> None: ...

def test_start_set_train_queue(mocker: pytest.MonkeyPatch, freqai_conf: dict, caplog: pytest.LogCaptureFixture) -> None: ...

def test_get_required_data_timerange(mocker: pytest.MonkeyPatch, freqai_conf: dict) -> TimeRange: ...

def test_download_all_data_for_training(mocker: pytest.MonkeyPatch, freqai_conf: dict, caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None: ...

@pytest.mark.usefixtures('init_persistence')
@pytest.mark.parametrize('dp_exists', [False, True])
def test_get_state_info(mocker: pytest.MonkeyPatch, freqai_conf: dict, dp_exists: bool, caplog: pytest.LogCaptureFixture, tickers: Any) -> None: ...