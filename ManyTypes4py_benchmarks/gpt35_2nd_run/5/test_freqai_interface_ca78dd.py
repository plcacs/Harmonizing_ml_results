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
from typing import List, Tuple

def can_run_model(model: str) -> None:
    is_pytorch_model = 'Reinforcement' in model or 'PyTorch' in model
    if is_arm() and 'Catboost' in model:
        pytest.skip('CatBoost is not supported on ARM.')
    if is_pytorch_model and is_mac():
        pytest.skip('Reinforcement learning / PyTorch module not available on intel based Mac OS.')

def test_extract_data_and_train_model_Standard(mocker, freqai_conf, model: str, pca: bool, dbscan: bool, float32: bool, can_short: bool, shuffle: bool, buffer: int, noise: int) -> None:
    ...

def test_extract_data_and_train_model_MultiTargets(mocker, freqai_conf, model: str, strat: str) -> None:
    ...

def test_extract_data_and_train_model_Classifiers(mocker, freqai_conf, model: str) -> None:
    ...

def test_start_backtesting(mocker, freqai_conf, model: str, num_files: int, strat: str, caplog) -> None:
    ...

def test_start_backtesting_subdaily_backtest_period(mocker, freqai_conf) -> None:
    ...

def test_start_backtesting_from_existing_folder(mocker, freqai_conf, caplog) -> None:
    ...

def test_backtesting_fit_live_predictions(mocker, freqai_conf, caplog) -> None:
    ...

def test_plot_feature_importance(mocker, freqai_conf) -> None:
    ...

def test_freqai_informative_pairs(mocker, freqai_conf, timeframes: List[str], corr_pairs: List[str]) -> None:
    ...

def test_start_set_train_queue(mocker, freqai_conf, caplog) -> None:
    ...

def test_get_required_data_timerange(mocker, freqai_conf) -> None:
    ...

def test_download_all_data_for_training(mocker, freqai_conf, caplog, tmp_path) -> None:
    ...

def test_get_state_info(mocker, freqai_conf, dp_exists: bool, caplog, tickers: Tuple[str, str]) -> None:
    ...
