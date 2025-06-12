import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import MagicMock

import pytest
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

def can_run_model(model: str) -> None:
    is_pytorch_model: bool = 'Reinforcement' in model or 'PyTorch' in model
    if is_arm() and 'Catboost' in model:
        pytest.skip('CatBoost is not supported on ARM.')
    if is_pytorch_model and is_mac():
        pytest.skip('Reinforcement learning / PyTorch module not available on intel based Mac OS.')

@pytest.mark.parametrize('model, pca, dbscan, float32, can_short, shuffle, buffer, noise', [
    ('LightGBMRegressor', True, False, True, True, False, 0, 0),
    ('XGBoostRegressor', False, True, False, True, False, 10, 0.05),
    ('XGBoostRFRegressor', False, False, False, True, False, 0, 0),
    ('CatboostRegressor', False, False, False, True, True, 0, 0),
    ('PyTorchMLPRegressor', False, False, False, False, False, 0, 0),
    ('PyTorchTransformerRegressor', False, False, False, False, False, 0, 0),
    ('ReinforcementLearner', False, True, False, True, False, 0, 0),
    ('ReinforcementLearner_multiproc', False, False, False, True, False, 0, 0),
    ('ReinforcementLearner_test_3ac', False, False, False, False, False, 0, 0),
    ('ReinforcementLearner_test_3ac', False, False, False, True, False, 0, 0),
    ('ReinforcementLearner_test_4ac', False, False, False, True, False, 0, 0)
])
def test_extract_data_and_train_model_Standard(
    mocker: Any,
    freqai_conf: Dict[str, Any],
    model: str,
    pca: bool,
    dbscan: bool,
    float32: bool,
    can_short: bool,
    shuffle: bool,
    buffer: int,
    noise: float
) -> None:
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
        pytorch_mlp_mtp: Dict[str, Any] = mock_pytorch_mlp_model_training_parameters()
        freqai_conf['freqai']['model_training_parameters'].update(pytorch_mlp_mtp)
        if 'Transformer' in model:
            freqai_conf.update({'conv_width': 10})
    strategy: Any = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange: Any = get_patched_exchange(mocker, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange)
    strategy.freqai_info = freqai_conf.get('freqai', {})
    freqai: Any = strategy.freqai
    freqai.live = True
    freqai.activate_tensorboard = test_tb
    freqai.can_short = can_short
    freqai.dk = FreqaiDataKitchen(freqai_conf)
    freqai.dk.live = True
    freqai.dk.set_paths('ADA/BTC', 10000)
    timerange: TimeRange = TimeRange.parse_timerange('20180110-20180130')
    freqai.dd.load_all_pair_histories(timerange, freqai.dk)
    freqai.dd.pair_dict = MagicMock()
    data_load_timerange: TimeRange = TimeRange.parse_timerange('20180125-20180130')
    new_timerange: TimeRange = TimeRange.parse_timerange('20180127-20180130')
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

@pytest.mark.parametrize('model, strat', [
    ('LightGBMRegressorMultiTarget', 'freqai_test_multimodel_strat'),
    ('XGBoostRegressorMultiTarget', 'freqai_test_multimodel_strat'),
    ('CatboostRegressorMultiTarget', 'freqai_test_multimodel_strat'),
    ('LightGBMClassifierMultiTarget', 'freqai_test_multimodel_classifier_strat'),
    ('CatboostClassifierMultiTarget', 'freqai_test_multimodel_classifier_strat')
])
def test_extract_data_and_train_model_MultiTargets(
    mocker: Any,
    freqai_conf: Dict[str, Any],
    model: str,
    strat: str
) -> None:
    can_run_model(model)
    freqai_conf.update({'timerange': '20180110-20180130'})
    freqai_conf.update({'strategy': strat})
    freqai_conf.update({'freqaimodel': model})
    strategy: Any = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange: Any = get_patched_exchange(mocker, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange)
    strategy.freqai_info = freqai_conf.get('freqai', {})
    freqai: Any = strategy.freqai
    freqai.live = True
    freqai.dk = FreqaiDataKitchen(freqai_conf)
    freqai.dk.live = True
    timerange: TimeRange = TimeRange.parse_timerange('20180110-20180130')
    freqai.dd.load_all_pair_histories(timerange, freqai.dk)
    freqai.dd.pair_dict = MagicMock()
    data_load_timerange: TimeRange = TimeRange.parse_timerange('20180110-20180130')
    new_timerange: TimeRange = TimeRange.parse_timerange('20180120-20180130')
    freqai.dk.set_paths('ADA/BTC', None)
    freqai.extract_data_and_train_model(new_timerange, 'ADA/BTC', strategy, freqai.dk, data_load_timerange)
    assert len(freqai.dk.label_list) == 2
    assert Path(freqai.dk.data_path / f'{freqai.dk.model_filename}_model.joblib').is_file()
    assert Path(freqai.dk.data_path / f'{freqai.dk.model_filename}_metadata.json').is_file()
    assert Path(freqai.dk.data_path / f'{freqai.dk.model_filename}_trained_df.pkl').is_file()
    assert len(freqai.dk.data['training_features_list']) == 14
    shutil.rmtree(Path(freqai.dk.full_path))

@pytest.mark.parametrize('model', [
    'LightGBMClassifier',
    'CatboostClassifier',
    'XGBoostClassifier',
    'XGBoostRFClassifier',
    'SKLearnRandomForestClassifier',
    'PyTorchMLPClassifier'
])
def test_extract_data_and_train_model_Classifiers(
    mocker: Any,
    freqai_conf: Dict[str, Any],
    model: str
) -> None:
    can_run_model(model)
    freqai_conf.update({'freqaimodel': model})
    freqai_conf.update({'strategy': 'freqai_test_classifier'})
    freqai_conf.update({'timerange': '20180110-20180130'})
    strategy: Any = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange: Any = get_patched_exchange(mocker, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange)
    strategy.freqai_info = freqai_conf.get('freqai', {})
    freqai: Any = strategy.freqai
    freqai.live = True
    freqai.dk = FreqaiDataKitchen(freqai_conf)
    freqai.dk.live = True
    timerange: TimeRange = TimeRange.parse_timerange('20180110-20180130')
    freqai.dd.load_all_pair_histories(timerange, freqai.dk)
    freqai.dd.pair_dict = MagicMock()
    data_load_timerange: TimeRange = TimeRange.parse_timerange('20180110-20180130')
    new_timerange: TimeRange = TimeRange.parse_timerange('20180120-20180130')
    freqai.dk.set_paths('ADA/BTC', None)
    freqai.extract_data_and_train_model(new_timerange, 'ADA/BTC', strategy, freqai.dk, data_load_timerange)
    if 'PyTorchMLPClassifier':
        pytorch_mlp_mtp: Dict[str, Any] = mock_pytorch_mlp_model_training_parameters()
        freqai_conf['freqai']['model_training_parameters'].update(pytorch_mlp_mtp)
    if freqai.dd.model_type == 'joblib':
        model_file_extension: str = '.joblib'
    elif freqai.dd.model_type == 'pytorch':
        model_file_extension = '.zip'
    else:
        raise Exception(f"Unsupported model type: {freqai.dd.model_type}, can't assign model_file_extension")
    assert Path(freqai.dk.data_path / f'{freqai.dk.model_filename}_model{model_file_extension}').exists()
    assert Path(freqai.dk.data_path / f'{freqai.dk.model_filename}_metadata.json').exists()
    assert Path(freqai.dk.data_path / f'{freqai.dk.model_filename}_trained_df.pkl').exists()
    shutil.rmtree(Path(freqai.dk.full_path))

@pytest.mark.parametrize('model, num_files, strat', [
    ('LightGBMRegressor', 2, 'freqai_test_strat'),
    ('XGBoostRegressor', 2, 'freqai_test_strat'),
    ('CatboostRegressor', 2, 'freqai_test_strat'),
    ('PyTorchMLPRegressor', 2, 'freqai_test_strat'),
    ('PyTorchTransformerRegressor', 2, 'freqai_test_strat'),
    ('ReinforcementLearner', 3, 'freqai_rl_test_strat'),
    ('XGBoostClassifier', 2, 'freqai_test_classifier'),
    ('LightGBMClassifier', 2, 'freqai_test_classifier'),
    ('CatboostClassifier', 2, 'freqai_test_classifier'),
    ('PyTorchMLPClassifier', 2, 'freqai_test_classifier')
])
def test_start_backtesting(
    mocker: Any,
    freqai_conf: Dict[str, Any],
    model: str,
    num_files: int,
    strat: str,
    caplog: Any
) -> None:
    can_run_model(model)
    test_tb: bool = True
    if is_mac() and (not is_arm()):
        test_tb = False
    freqai_conf.get('freqai', {}).update({'save_backtest_models': True})
    freqai_conf['runmode'] = RunMode.BACKTEST
    Trade.use_db = False
    freqai_conf.update({'freqaimodel': model})
    freqai_conf.update({'timerange': '20180120-20180130'})
    freqai_conf.update({'strategy': strat})
    if 'ReinforcementLearner' in model:
        freqai_conf = make_rl_config(freqai_conf)
    if 'test_4ac' in model:
        freqai_conf['freqaimodel_path'] = str(Path(__file__).parents[1] / 'freqai' / 'test_models')
    if 'PyTorch' in model:
        pytorch_mlp_mtp: Dict[str, Any] = mock_pytorch_mlp_model_training_parameters()
        freqai_conf['freqai']['model_training_parameters'].update(pytorch_mlp_mtp)
        if 'Transformer' in model:
            freqai_conf.update({'conv_width': 10})
    freqai_conf.get('freqai', {}).get('feature_parameters', {}).update({'indicator_periods_candles': [2]})
    strategy: Any = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange: Any = get_patched_exchange(mocker, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange)
    strategy.freqai_info = freqai_conf.get('freqai', {})
    freqai: Any = strategy.freqai
    freqai.live = False
    freqai.activate_tensorboard = test_tb
    freqai.dk = FreqaiDataKitchen(freqai_conf)
    timerange: TimeRange = TimeRange.parse_timerange('20180110-20180130')
    freqai.dd.load_all_pair_histories(timerange, freqai.dk)
    sub_timerange: TimeRange = TimeRange.parse_timerange('20180110-20180130')
    _, base_df = freqai.dd.get_base_and_corr_dataframes(sub_timerange, 'LTC/BTC', freqai.dk)
    df = base_df[freqai_conf['timeframe']]
    metadata: Dict[str, str] = {'pair': 'LTC/BTC'}
    freqai.dk.set_paths('LTC/BTC', None)
    freqai.start_backtesting(df, metadata, freqai.dk, strategy)
    model_folders: List[Path] = [x for x in freqai.dd.full_path.iterdir() if x.is_dir()]
    assert len(model_folders) == num_files
    Trade.use_db = True
    Backtesting.cleanup()
    shutil.rmtree(Path(freqai.dk.full_path))

def test_start_backtesting_subdaily_backtest_period(mocker: Any, freqai_conf: Dict[str, Any]) -> None:
    freqai_conf.update({'timerange': '20180120-20180124'})
    freqai_conf['runmode'] = 'backtest'
    freqai_conf.get('freqai', {}).update({'backtest_period_days': 0.5, 'save_backtest_models': True})
    freqai_conf.get('freqai', {}).get('feature_parameters', {}).update({'indicator_periods_candles': [2]})
    strategy: Any = get_patched_freqai_strategy(mocker, freqai_conf)
    exchange: Any = get_patched_exchange(mocker, freqai_conf)
    strategy.dp = DataProvider(freqai_conf, exchange)
    strategy.freq