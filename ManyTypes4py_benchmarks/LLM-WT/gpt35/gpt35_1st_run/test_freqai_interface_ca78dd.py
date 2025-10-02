from freqtrade.configuration import TimeRange
from freqtrade.data.dataprovider import DataProvider
from freqtrade.enums import RunMode
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.utils import download_all_data_for_training, get_required_data_timerange
from freqtrade.optimize.backtesting import Backtesting
from freqtrade.persistence import Trade
from freqtrade.plugins.pairlistmanager import PairListManager
from pathlib import Path
from tests.conftest import EXMS, create_mock_trades, get_patched_exchange, is_arm, is_mac, log_has_re
from tests.freqai.conftest import get_patched_freqai_strategy, make_rl_config, mock_pytorch_mlp_model_training_parameters
from unittest.mock import MagicMock
import logging
import pytest
import shutil

def can_run_model(model: str) -> None:
    is_pytorch_model = 'Reinforcement' in model or 'PyTorch' in model
    if is_arm() and 'Catboost' in model:
        pytest.skip('CatBoost is not supported on ARM.')
    if is_pytorch_model and is_mac():
        pytest.skip('Reinforcement learning / PyTorch module not available on intel based Mac OS.')

@pytest.mark.parametrize('model, pca, dbscan, float32, can_short, shuffle, buffer, noise', [('LightGBMRegressor', True, False, True, True, False, 0, 0), ('XGBoostRegressor', False, True, False, True, False, 10, 0.05), ('XGBoostRFRegressor', False, False, False, True, False, 0, 0), ('CatboostRegressor', False, False, False, True, True, 0, 0), ('PyTorchMLPRegressor', False, False, False, False, False, 0, 0), ('PyTorchTransformerRegressor', False, False, False, False, False, 0, 0), ('ReinforcementLearner', False, True, False, True, False, 0, 0), ('ReinforcementLearner_multiproc', False, False, False, True, False, 0, 0), ('ReinforcementLearner_test_3ac', False, False, False, False, False, 0, 0), ('ReinforcementLearner_test_3ac', False, False, False, True, False, 0, 0), ('ReinforcementLearner_test_4ac', False, False, False, True, False, 0, 0)])
def test_extract_data_and_train_model_Standard(mocker, freqai_conf: dict, model: str, pca: bool, dbscan: bool, float32: bool, can_short: bool, shuffle: bool, buffer: int, noise: int) -> None:
    can_run_model(model)
    test_tb: bool = True
    if is_mac():
        test_tb = False
    model_save_ext: str = 'joblib'
    freqai_conf.update({'freqaimodel': model})
    freqai_conf.update({'timerange': '20180110-20180130'})
    freqai_conf.update({'strategy': 'freqai_test_strat'})
    freqai_conf['freqai']['feature_parameters'].update({'principal_component_analysis': pca})
    freqai_conf['freqai']['feature_parameters'].update({'use_DBSCAN_to_remove_outliers': dbscan})
    freqai_conf.update({'reduce_df_footprint': float32})
    freqai_conf['freqai']['feature_parameters'].update({'shuffle_after_split': shuffle})
    freqai_conf['freqai']['feature_parameters'].update({'buffer_train_data_candles': buffer})
    freqai_conf['freqai']['feature_parameters'].update({'noise_standard_deviation': noise})
    if 'ReinforcementLearner' in model:
        model_save_ext = 'zip'
        freqai_conf = make_rl_config(freqai_conf)
        freqai_conf['freqai']['feature_parameters'].update({'use_SVM_to_remove_outliers': True})
        freqai_conf['freqai']['feature_parameters'].update({'DI_threshold': 2})
        freqai_conf['freqai']['data_split_parameters'].update({'shuffle': True})
    if 'test_3ac' in model or 'test_4ac' in model:
        freqai_conf['freqaimodel_path'] = str(Path(__file__).parents[1] / 'freqai' / 'test_models')
        freqai_conf['freqai']['rl_config']['drop_ohlc_from_features'] = True
    if 'PyTorch' in model:
        model_save_ext = 'zip'
        pytorch_mlp_mtp = mock_pytorch_mlp_model_training_parameters()
        freqai_conf['freqai']['model_training_parameters'].update(pytorch_mlp_mtp)
        if 'Transformer' in model:
            freqai_conf.update({'conv_width': 10})
    strategy = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange = get_patched_exchange(mocker, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange)
    strategy.freqai_info = freqai_conf.get('freqai', {})
    freqai = strategy.freqai
    freqai.live = True
    freqai.activate_tensorboard = test_tb
    freqai.can_short = can_short
    freqai.dk = FreqaiDataKitchen(freqai_conf)
    freqai.dk.live = True
    freqai.dk.set_paths('ADA/BTC', 10000)
    timerange = TimeRange.parse_timerange('20180110-20180130')
    freqai.dd.load_all_pair_histories(timerange, freqai.dk)
    freqai.dd.pair_dict = MagicMock()
    data_load_timerange = TimeRange.parse_timerange('20180125-20180130')
    new_timerange = TimeRange.parse_timerange('20180127-20180130')
    freqai.dk.set_paths('ADA/BTC', None)
    freqai.train_timer('start', 'ADA/BTC')
    freqai.extract_data_and_train_model(new_timerange, 'ADA/BTC', strategy, freqai.dk, data_load_timerange)
    freqai.train_timer('stop', 'ADA/BTC')
    freqai.dd.save_metric_tracker_to_disk()
    freqai.dd.save_drawer_to_disk()
    assert Path(freqai.dk.full_path / 'metric_tracker.json').is_file()
    assert Path(freqai.dk.full_path / 'pair_dictionary.json').is_file()
    assert Path(freqai.dk.data_path / f'{freqai.dk.model_filename}_model.{model_save_ext}').is_file()
    assert Path(freqai.dk.data_path / f'{freqai.dk.model_filename}_metadata.json').is_file()
    assert Path(freqai.dk.data_path / f'{freqai.dk.model_filename}_trained_df.pkl').is_file()
    shutil.rmtree(Path(freqai.dk.full_path))

# Add type annotations for other test functions as well
