import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import datasieve.transforms as ds
import numpy as np
import pandas as pd
import psutil
from datasieve.pipeline import Pipeline
from datasieve.transforms import SKLearnWrapper
from numpy.typing import NDArray
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from freqtrade.configuration import TimeRange
from freqtrade.constants import DOCS_LINK, Config
from freqtrade.data.dataprovider import DataProvider
from freqtrade.enums import RunMode
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import timeframe_to_seconds
from freqtrade.freqai.data_drawer import FreqaiDataDrawer
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.utils import get_tb_logger, plot_feature_importance, record_params
from freqtrade.strategy.interface import IStrategy

pd.options.mode.chained_assignment = None
logger = logging.getLogger(__name__)

class IFreqaiModel(ABC):
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config: Dict[str, Any] = config
        self.assert_config(self.config)
        self.freqai_info: Dict[str, Any] = config['freqai']
        self.data_split_parameters: Dict[str, Any] = config.get('freqai', {}).get('data_split_parameters', {})
        self.model_training_parameters: Dict[str, Any] = config.get('freqai', {}).get('model_training_parameters', {})
        self.identifier: str = self.freqai_info.get('identifier', 'no_id_provided')
        self.retrain: bool = False
        self.first: bool = True
        self.set_full_path()
        self.save_backtest_models: bool = self.freqai_info.get('save_backtest_models', True)
        if self.save_backtest_models:
            logger.info('Backtesting module configured to save all models.')
        self.dd: FreqaiDataDrawer = FreqaiDataDrawer(Path(self.full_path), self.config)
        self.current_candle: datetime = datetime.fromtimestamp(637887600, tz=timezone.utc)
        self.dd.current_candle = self.current_candle
        self.scanning: bool = False
        self.ft_params: Dict[str, Any] = self.freqai_info['feature_parameters']
        self.corr_pairlist: List[str] = self.ft_params.get('include_corr_pairlist', [])
        self.keras: bool = self.freqai_info.get('keras', False)
        if self.keras and self.ft_params.get('DI_threshold', 0):
            self.ft_params['DI_threshold'] = 0
            logger.warning('DI threshold is not configured for Keras models yet. Deactivating.')
        self.CONV_WIDTH: int = self.freqai_info.get('conv_width', 1)
        self.class_names: List[str] = []
        self.pair_it: int = 0
        self.pair_it_train: int = 0
        self.total_pairs: int = len(self.config.get('exchange', {}).get('pair_whitelist'))
        self.train_queue: deque = self._set_train_queue()
        self.inference_time: float = 0
        self.train_time: float = 0
        self.begin_time: float = 0
        self.begin_time_train: float = 0
        self.base_tf_seconds: int = timeframe_to_seconds(self.config['timeframe'])
        self.continual_learning: bool = self.freqai_info.get('continual_learning', False)
        self.plot_features: int = self.ft_params.get('plot_feature_importances', 0)
        self.corr_dataframes: Dict[str, DataFrame] = {}
        self.get_corr_dataframes: bool = True
        self._threads: List[threading.Thread] = []
        self._stop_event: threading.Event = threading.Event()
        self.metadata: Dict[str, Any] = self.dd.load_global_metadata_from_disk()
        self.data_provider: Optional[DataProvider] = None
        self.max_system_threads: int = max(int(psutil.cpu_count() * 2 - 2), 1)
        self.can_short: bool = True
        self.model: Optional[Any] = None
        if self.ft_params.get('principal_component_analysis', False) and self.continual_learning:
            self.ft_params.update({'principal_component_analysis': False})
            logger.warning('User tried to use PCA with continual learning. Deactivating PCA.')
        self.activate_tensorboard: bool = self.freqai_info.get('activate_tensorboard', True)
        record_params(config, self.full_path)

    def __getstate__(self) -> Dict[str, Any]:
        return {}

    def assert_config(self, config: Dict[str, Any]) -> None:
        if not config.get('freqai', {}):
            raise OperationalException('No freqai parameters found in configuration file.')

    def start(self, dataframe: DataFrame, metadata: Dict[str, Any], strategy: IStrategy) -> DataFrame:
        self.live: bool = strategy.dp.runmode in (RunMode.DRY_RUN, RunMode.LIVE)
        self.dd.set_pair_dict_info(metadata)
        self.data_provider = strategy.dp
        self.can_short = strategy.can_short
        if self.live:
            self.inference_timer('start')
            self.dk: FreqaiDataKitchen = FreqaiDataKitchen(self.config, self.live, metadata['pair'])
            dk: FreqaiDataKitchen = self.start_live(dataframe, metadata, strategy, self.dk)
            dataframe = dk.remove_features_from_df(dk.return_dataframe)
        else:
            self.dk = FreqaiDataKitchen(self.config, self.live, metadata['pair'])
            if not self.config.get('freqai_backtest_live_models', False):
                logger.info(f'Training {len(self.dk.training_timeranges)} timeranges')
                dk = self.start_backtesting(dataframe, metadata, self.dk, strategy)
                dataframe = dk.remove_features_from_df(dk.return_dataframe)
            else:
                logger.info('Backtesting using historic predictions (live models)')
                dk = self.start_backtesting_from_historic_predictions(dataframe, metadata, self.dk)
                dataframe = dk.return_dataframe
        self.clean_up()
        if self.live:
            self.inference_timer('stop', metadata['pair'])
        return dataframe

    def clean_up(self) -> None:
        self.model = None
        self.dk = None

    def _on_stop(self) -> None:
        self.dd.save_historic_predictions_to_disk()
        return

    def shutdown(self) -> None:
        logger.info('Stopping FreqAI')
        self._stop_event.set()
        self.data_provider = None
        self._on_stop()
        if self.freqai_info.get('wait_for_training_iteration_on_reload', True):
            logger.info('Waiting on Training iteration')
            for _thread in self._threads:
                _thread.join()
        else:
            logger.warning('Breaking current training iteration because you set wait_for_training_iteration_on_reload to  False.')

    def start_scanning(self, *args: Any, **kwargs: Any) -> None:
        _thread: threading.Thread = threading.Thread(target=self._start_scanning, args=args, kwargs=kwargs)
        self._threads.append(_thread)
        _thread.start()

    def _start_scanning(self, strategy: IStrategy) -> None:
        while not self._stop_event.is_set():
            time.sleep(1)
            pair: str = self.train_queue[0]
            if pair not in strategy.dp.current_whitelist():
                self.train_queue.popleft()
                logger.warning(f'{pair} not in current whitelist, removing from train queue.')
                continue
            _, trained_timestamp = self.dd.get_pair_dict_info(pair)
            dk: FreqaiDataKitchen = FreqaiDataKitchen(self.config, self.live, pair)
            retrain, new_trained_timerange, data_load_timerange = dk.check_if_new_training_required(trained_timestamp)
            if retrain:
                self.train_timer('start')
                dk.set_paths(pair, new_trained_timerange.stopts)
                try:
                    self.extract_data_and_train_model(new_trained_timerange, pair, strategy, dk, data_load_timerange)
                except Exception as msg:
                    logger.exception(f'Training {pair} raised exception {msg.__class__.__name__}. Message: {msg}, skipping.')
                self.train_timer('stop', pair)
                self.train_queue.rotate(-1)
                self.dd.save_historic_predictions_to_disk()
                if self.freqai_info.get('write_metrics_to_disk', False):
                    self.dd.save_metric_tracker_to_disk()

    def start_backtesting(self, dataframe: DataFrame, metadata: Dict[str, Any], dk: FreqaiDataKitchen, strategy: IStrategy) -> FreqaiDataKitchen:
        self.pair_it += 1
        train_it: int = 0
        pair: str = metadata['pair']
        populate_indicators: bool = True
        check_features: bool = True
        for tr_train, tr_backtest in zip(dk.training_timeranges, dk.backtesting_timeranges, strict=False):
            _, _ = self.dd.get_pair_dict_info(pair)
            train_it += 1
            total_trains: int = len(dk.backtesting_timeranges)
            self.training_timerange: TimeRange = tr_train
            len_backtest_df: int = len(dataframe.loc[(dataframe['date'] >= tr_backtest.startdt) & (dataframe['date'] < tr_backtest.stopdt), :])
            if not self.ensure_data_exists(len_backtest_df, tr_backtest, pair):
                continue
            self.log_backtesting_progress(tr_train, pair, train_it, total_trains)
            timestamp_model_id: int = int(tr_train.stopts)
            if dk.backtest_live_models:
                timestamp_model_id = int(tr_backtest.startts)
            dk.set_paths(pair, timestamp_model_id)
            dk.set_new_model_names(pair, timestamp_model_id)
            if dk.check_if_backtest_prediction_is_valid(len_backtest_df):
                if check_features:
                    self.dd.load_metadata(dk)
                    df_fts: DataFrame = self.dk.use_strategy_to_populate_indicators(strategy, prediction_dataframe=dataframe.tail(1), pair=pair)
                    df_fts = dk.remove_special_chars_from_feature_names(df_fts)
                    dk.find_features(df_fts)
                    self.check_if_feature_list_matches_strategy(dk)
                    check_features = False
                append_df: DataFrame = dk.get_backtesting_prediction()
                dk.append_predictions(append_df)
            else:
                if populate_indicators:
                    dataframe = self.dk.use_strategy_to_populate_indicators(strategy, prediction_dataframe=dataframe, pair=pair)
                    populate_indicators = False
                dataframe_base_train: DataFrame = dataframe.loc[dataframe['date'] < tr_train.stopdt, :]
                dataframe_base_train = strategy.set_freqai_targets(dataframe_base_train, metadata=metadata)
                dataframe_base_backtest: DataFrame = dataframe.loc[dataframe['date'] < tr_backtest.stopdt, :]
                dataframe_base_backtest = strategy.set_freqai_targets(dataframe_base_backtest, metadata=metadata)
                tr_train = dk.buffer_timerange(tr_train)
                dataframe_train: DataFrame = dk.slice_dataframe(tr_train, dataframe_base_train)
                dataframe_backtest: DataFrame = dk.slice_dataframe(tr_backtest, dataframe_base_backtest)
                dataframe_train = dk.remove_special_chars_from_feature_names(dataframe_train)
                dataframe_backtest = dk.remove_special_chars_from_feature_names(dataframe_backtest)
                dk.get_unique_classes_from_labels(dataframe_train)
                if not self.model_exists(dk):
                    dk.find_features(dataframe_train)
                    dk.find_labels(dataframe_train)
                    try:
                        self.tb_logger = get_tb_logger(self.dd.model_type, dk.data_path, self.activate_tensorboard)
                        self.model = self.train(dataframe_train, pair, dk)
                        self.tb_logger.close()
                    except Exception as msg:
                        logger.warning(f'Training {pair} raised exception {msg.__class__.__name__}. Message: {msg}, skipping.', exc_info=True)
                        self.model = None
                    self.dd.pair_dict[pair]['trained_timestamp'] = int(tr_train.stopts)
                    if self.plot_features and self.model is not None:
                        plot_feature_importance(self.model, pair, dk, self.plot_features)
                    if self.save_backtest_models and self.model is not None:
                        logger.info('Saving backtest model to disk.')
                        self.dd.save_data(self.model, pair, dk)
                    else:
                        logger.info('Saving metadata to disk.')
                        self.dd.save_metadata(dk)
                else:
                    self.model = self.dd.load_data(pair, dk)
                pred_df, do_preds = self.predict(dataframe_backtest, dk)
                append_df = dk.get_predictions_to_append(pred_df, do_preds, dataframe_backtest)
                dk.append_predictions(append_df)
                dk.save_backtesting_prediction(append_df)
        self.backtesting_fit_live_predictions(dk)
        dk.fill_predictions(dataframe)
        return dk

    def start_live(self, dataframe: DataFrame, metadata: Dict[str, Any], strategy: IStrategy, dk: FreqaiDataKitchen) -> FreqaiDataKitchen:
        if not strategy.process_only_new_candles:
            raise OperationalException('You are trying to use a FreqAI strategy with process_only_new_candles = False. This is not supported by FreqAI, and it is therefore aborting.')
        _, trained_timestamp = self.dd.get_pair_dict_info(metadata['pair'])
        if self.dd.historic_data:
            self.dd.update_historic_data(strategy, dk)
            logger.debug(f'Updating historic data on pair {metadata["pair"]}')
            self.track_current_candle()
        _, new_trained_timerange, data_load_timerange = dk.check_if_new_training_required(trained_timestamp)
        dk.set_paths(metadata['pair'], new_trained_timerange.stopts)
        if not self.dd.historic_data:
            self.dd.load_all_pair_histories(data_load_timerange, dk)
        if not self.scanning:
            self.scanning = True
            self.start_scanning(strategy)
        self.model = self.dd.load_data(metadata['pair'], dk)
        dataframe = dk.use_strategy_to_populate_indicators(strategy, prediction_dataframe=dataframe, pair=metadata['pair'], do_corr_pairs=self.get_corr_dataframes)
        if not self.model:
            logger.warning(f'No model ready for {metadata["pair"]}, returning null values to strategy.')
            self.dd.return_null_values_to_strategy(dataframe, dk)
            return dk
        if self.corr_pairlist:
            dataframe = self.cache_corr_pairlist_dfs(dataframe, dk)
        dk.find_labels(dataframe)
        self.build_strategy_return_arrays(dataframe, dk, metadata['pair'], trained_timestamp)
        return dk

    def build_strategy_return_arrays(self, dataframe: DataFrame, dk: FreqaiDataKitchen, pair: str, trained_timestamp: int) -> None:
        if pair not in self.dd.model_return_values:
            pred_df, do_preds = self.predict(dataframe, dk)
            if pair not in self.dd.historic_predictions:
                self.set_initial_historic_predictions(pred_df, dk, pair, dataframe)
            self.dd.set_initial_return_values(pair, pred_df, dataframe)
            dk.return_dataframe = self.dd.attach_return_values_to_return_dataframe(pair, dataframe)
            return
        elif self.dk.check_if_model_expired(trained_timestamp):
            pred_df = DataFrame(np.zeros((2, len(dk.label_list))), columns=dk.label_list)
            do_preds = np.ones(2, dtype=np.int_) * 2
            dk.DI_values = np.zeros(2)
            logger.warning(f'Model expired for {pair}, returning null values to strategy. Strategy construction should take care to consider this event with prediction == 0 and do_predict == 2')
        else:
            pred_df, do_preds = self.predict(dataframe.iloc[-self.CONV_WIDTH:], dk, first=False)
        if self.freqai_info.get('fit_live_predictions_candles', 0) and self.live:
            self.fit_live_predictions(dk, pair)
        self.dd.append_model_predictions(pair, pred_df, do_preds, dk, dataframe)
        dk.return_dataframe = self.dd.attach_return_values_to_return_dataframe(pair, dataframe)
        return

    def check_if_feature_list_matches_strategy(self, dk: FreqaiDataKitchen) -> None:
        if 'training_features_list_raw' in dk.data:
            feature_list = dk.data['training_features_list_raw']
        else:
            feature_list = dk.data['training_features_list']
        if dk.training_features_list != feature_list:
            raise OperationalException('Trying to access pretrained model with `identifier` but found different features furnished by current strategy. Change `identifier` to train from scratch, or ensure the strategy is furnishing the same features as the pretrained model. In case of --strategy-list, please be aware that FreqAI requires all strategies to maintain identical feature_engineering_* functions')

    def define_data_pipeline(self, threads: int = -1) -> Pipeline:
        ft_params = self.freqai_info['feature_parameters']
        pipe_steps: List[Tuple[str, Any]] = [('const', ds.VarianceThreshold(threshold=0)), ('scaler', SKLearnWrapper(MinMaxScaler(feature_range=(-1, 1))))]
        if ft_params.get('principal_component_analysis', False):
            pipe_steps.append(('pca', ds.PCA(n_components=0.999)))
            pipe_steps.append(('post-pca-scaler', SKLearnWrapper(MinMaxScaler(feature_range=(-1, 1)))))
        if ft_params.get('use_SVM_to_remove_outliers', False):
            svm_params = ft_params.get('svm_params', {'shuffle': False, 'nu': 0.01})
            pipe_steps.append(('svm', ds.SVMOutlierExtractor(**svm_params)))
        di: int = ft_params.get('DI_threshold', 0)
        if di:
            pipe_steps.append(('di', ds.DissimilarityIndex(di_threshold=di, n_jobs=threads)))
        if ft_params.get('use_DBSCAN_to_remove_outliers', False):
            pipe_steps.append(('dbscan', ds.DBSCAN(n_jobs=threads)))
        sigma: float = self.freqai_info['feature_parameters'].get('noise_standard_deviation', 0)
        if sigma:
            pipe_steps.append(('noise', ds.Noise(sigma=sigma)))
        return Pipeline(pipe_steps)

    def define_label_pipeline(self, threads: int = -1) -> Pipeline:
        label_pipeline: Pipeline = Pipeline([('scaler', SKLearnWrapper(MinMaxScaler(feature_range=(-1, 1))))])
        return label_pipeline

    def model_exists(self, dk: FreqaiDataKitchen) -> bool:
        if self.dd.model_type == 'joblib':
            file_type: str = '.joblib'
        elif self.dd.model_type in ['stable_baselines3', 'sb3_contrib', 'pytorch']:
            file_type = '.zip'
        path_to_modelfile: Path = Path(dk.data_path / f'{dk.model_filename}_model{file_type}')
        file_exists: bool = path_to_modelfile.is_file()
        if file_exists:
            logger.info('Found model at %s', dk.data_path / dk.model_filename)
        else:
            logger.info('Could not find model at %s', dk.data_path / dk.model_filename)
        return file_exists

    def set_full_path(self) -> None:
        self.full_path: Path = Path(self.config['user_data_dir'] / 'models' / f'{self.identifier}')
        self.full_path.mkdir(parents=True, exist_ok=True)

    def extract_data_and_train_model(self, new_trained_timerange: TimeRange, pair: str, strategy: IStrategy, dk: FreqaiDataKitchen, data_load_timerange: TimeRange) -> None:
        corr_dataframes, base_dataframes = self.dd.get_base_and_corr_dataframes(data_load_timerange, pair, dk)
        unfiltered_dataframe: DataFrame = dk.use_strategy_to_populate_indicators(strategy, corr_dataframes, base_dataframes, pair)
        trained_timestamp: int = new_trained_timerange.stopts
        buffered_timerange: TimeRange = dk.buffer_timerange(new_trained_timerange)
        unfiltered_dataframe = dk.slice_dataframe(buffered_timerange, unfiltered_dataframe)
        dk.find_features(unfiltered_dataframe)
        dk.find_labels(unfiltered_dataframe)
        self.tb_logger = get_tb_logger(self.dd.model_type, dk.data_path, self.activate_tensorboard)
        model = self.train(unfiltered_dataframe, pair, dk)
        self.tb_logger.close()
        self.dd.pair_dict[pair]['trained_timestamp'] = trained_timestamp
        dk.set_new_model_names(pair, trained_timestamp)
        self.dd.save_data(model, pair, dk)
        if self.plot_features:
            plot_feature_importance(model, pair, dk, self.plot_features)
        self.dd.purge_old_models()

    def set_initial_historic_predictions(self, pred_df: DataFrame, dk: FreqaiDataKitchen, pair: str, strat_df: DataFrame) -> None:
        self.dd.historic_predictions[pair] = pred_df
        hist_preds_df: DataFrame = self.dd.historic_predictions[pair]
        self.set_start_dry_live_date(strat_df)
        for label in hist_preds_df.columns:
            if hist_preds_df[label].dtype == object:
                continue
            hist_preds_df[f'{label}_mean'] = 0
            hist_preds_df[f'{label}_std'] = 0
        hist_preds_df['do_predict'] = 0
        if self.freqai_info['feature_parameters'].get('DI_threshold', 0) > 0:
            hist_preds_df['DI_values'] = 0
        for return_str in dk.data['extra_returns_per_train']:
            hist_preds_df[return_str] = dk.data['extra_returns_per_train'][return_str]
        hist_preds_df['high_price'] = strat_df['high']
        hist_preds_df['low_price'] = strat_df['low']
        hist_preds_df['close_price'] = strat_df['close']
        hist_preds_df['date_pred'] = strat_df['date']

    def fit_live_predictions(self, dk: FreqaiDataKitchen, pair: str) -> None:
        import scipy as spy
        full_labels: List[str] = dk.label_list + dk.unique_class_list
        num_candles: int = self.freqai_info.get('fit_live_predictions_candles', 100)
        dk.data['labels_mean'], dk.data['labels_std'] = ({}, {})
        for label in full_labels:
            if self.dd.historic_predictions[dk.pair][label].dtype == object:
                continue
            f: Tuple[float, float] = spy.stats.norm.fit(self.dd.historic_predictions[dk.pair][label].tail(num_candles))
            dk.data['labels_mean'][label], dk.data['labels_std'][label] = (f[0], f[1])
        return

    def inference_timer(self, do: str = 'start', pair: str = '') -> None:
        if do == 'start':
            self.pair_it += 1
            self.begin_time = time.time()
        elif do == 'stop':
            end: float = time.time()
            time_spent: float = end - self.begin_time
            if self.freqai_info.get('write_metrics_to_disk', False):
                self.dd.update_metric_tracker('inference_time', time_spent, pair)
            self.inference_time += time_spent
            if self.pair_it == self.total_pairs:
                logger.info(f'Total time spent inferencing pairlist {self.inference_time:.2f} seconds')
                self.pair_it = 0
                self.inference_time = 0
        return

    def train_timer(self, do: str = 'start', pair: str = '') -> None:
        if do == 'start':
            self.pair_it_train += 1
            self.begin_time_train = time.time()
        elif do == 'stop':
            end: float = time.time()
            time_spent: float = end - self.begin_time_train
            if self.freqai_info.get('write_metrics_to_disk', False):
                self.dd.collect_metrics(time_spent, pair)
            self.train_time += time_spent
            if self.pair_it_train == self.total_pairs:
                logger.info(f'Total time spent training pairlist {self.train_time:.2f} seconds')
                self.pair_it_train = 0
                self.train_time = 0
        return

    def get_init_model(self, pair: str) -> Optional[Any]:
        if pair not in self.dd.model_dictionary or not self.continual_learning:
            init_model = None
        else:
            init_model = self.dd.model_dictionary[pair]
        return init_model

    def _set_train_queue(self) -> deque:
        current_pairlist: List[str] = self.config.get('exchange', {}).get('pair_whitelist')
        if not self.dd.pair_dict:
            logger.info(f'Set fresh train queue from whitelist. Queue: {current_pairlist}')
            return deque(current_pairlist)
        best_queue: deque = deque()
        pair_dict_sorted: List[Tuple[str, Dict[str, Any]]] = sorted(self.dd.pair_dict.items(), key=lambda k: k[1]['trained_timestamp'])
        for pair in pair_dict_sorted:
            if pair[0] in current_pairlist:
                best_queue.append(pair[0])
        for pair in current_pairlist:
            if pair not in best_queue:
                best_queue.appendleft(pair)
        logger.info(f'Set existing queue from trained timestamps. Best approximation queue: {best_queue}')
        return best_queue

    def cache_corr_pairlist_dfs(self, dataframe: DataFrame, dk: FreqaiDataKitchen) -> DataFrame:
        if self.get_corr_dataframes:
            self.corr_dataframes = dk.extract_corr_pair_columns_from_populated_indicators(dataframe)
            if not self.corr_dataframes:
                logger.warning("Couldn't cache corr_pair dataframes for improved performance. Consider ensuring that the full coin/stake, e.g. XYZ/USD, is included in the column names when you are creating features in `feature_engineering_*` functions.")
            self.get_corr_dataframes = not bool(self.corr_dataframes)
        elif self.corr_dataframes:
            dataframe = dk.attach_corr_pair_columns(dataframe, self.corr_dataframes, dk.pair)
        return dataframe

    def track_current_candle(self) -> None:
        if self.dd.current_candle > self.current_candle:
            self.get_corr_dataframes = True
            self.pair_it = 1
            self.current_candle = self.dd.current_candle

    def ensure_data_exists(self, len_dataframe_backtest: int, tr_backtest: TimeRange, pair: str) -> bool:
        if self.config.get('freqai_backtest_live_models', False) and len_dataframe_backtest == 0:
            logger.info(f'No data found for pair {pair} from from {tr_backtest.start_fmt} to {tr_backtest.stop_fmt}. Probably more than one training within the same candle period.')
            return False
        return True

    def log_backtesting_progress(self, tr_train: TimeRange, pair: str, train_it: int, total_trains: int) -> None:
        if not self.config.get('freqai_backtest_live_models', False):
            logger.info(f'Training {pair}, {self.pair_it}/{self.total_pairs} pairs from {tr_train.start_fmt} to {tr_train.stop_fmt}, {train_it}/{total_trains} trains')

    def backtesting_fit_live_predictions(self, dk: FreqaiDataKitchen) -> None:
        fit_live_predictions_candles: int = self.freqai_info.get('fit_live_predictions_candles', 0)
        if fit_live_predictions_candles:
            logger.info('Applying fit_live_predictions in backtesting')
            label_columns: List[str] = [col for col in dk.full_df.columns if col.startswith('&') and (not (col.startswith('&') and col.endswith('_mean'))) and (not (col.startswith('&') and col.endswith('_std'))) and (col not in self.dk.data['extra_returns_per_train'])]
            for index in range(len(dk.full_df)):
                if index >= fit_live_predictions_candles:
                    self.dd.historic_predictions[self.dk.pair] = dk.full_df.iloc[index - fit_live_predictions_candles:index]
                    self.fit_live_predictions(self.dk, self.dk.pair)
                    for label in label_columns:
                        if dk.full_df[label].dtype == object:
                            continue
                        if 'labels_mean' in self.dk.data:
                            dk.full_df.at[index, f'{label}_mean'] = self.dk.data['labels_mean'][label]
                        if 'labels_std' in self.dk.data:
                            dk.full_df.at[index, f'{label}_std'] = self.dk.data['labels_std'][label]
                    for extra_col in self.dk.data['extra_returns_per_train']:
                        dk.full_df.at[index, f'{extra_col}'] = self.dk.data['extra_returns_per_train'][extra_col]
        return

    def update_metadata(self, metadata: Dict[str, Any]) -> None:
        self.dd.save_global_metadata_to_disk(metadata)
        self.metadata = metadata

    def set_start_dry_live_date(self, live_dataframe: DataFrame) -> None:
        key_name: str = 'start_dry_live_date'
        if key_name not in self.metadata:
            metadata: Dict[str, Any] = self.metadata
            metadata[key_name] = int(pd.to_datetime(live_dataframe.tail(1)['date'].values[0]).timestamp())
            self.update_metadata(metadata)

    def start_backtesting_from_historic_predictions(self, dataframe: DataFrame, metadata: Dict[str, Any], dk: FreqaiDataKitchen) -> FreqaiDataKitchen:
        pair: str = metadata['pair']
        dk.return_dataframe = dataframe
        saved_dataframe: DataFrame = self.dd.historic_predictions[pair]
        columns_to_drop: List[str] = list(set(saved_dataframe.columns).intersection(dk.return_dataframe.columns))
        dk.return_dataframe = dk.return_dataframe.drop(columns=list(columns_to_drop))
        dk.return_dataframe = pd.merge(dk.return_dataframe, saved_dataframe, how='left', left_on='date', right_on='date_pred')
        return dk

    @abstractmethod
    def train(self, unfiltered_df: DataFrame, pair: str, dk: FreqaiDataKitchen, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def fit(self, data_dictionary: Dict[str, Any], dk: FreqaiDataKitchen, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def predict(self, unfiltered_df: DataFrame, dk: FreqaiDataKitchen, **kwargs: Any) -> Tuple[NDArray, NDArray]:
        pass

    def data_cleaning_train(self, dk: FreqaiDataKitchen, pair: str) -> None:
        logger.warning(f'Your model {self.__class__.__name__} relies on the deprecated data pipeline. Please update your model to use the new data pipeline. This can be achieved by following the migration guide at {DOCS_LINK}/strategy_migration/#freqai-new-data-pipeline')
        dk.feature_pipeline = self.define_data_pipeline(threads=dk.thread_count)
        dd = dk.data_dictionary
        dd['train_features'], dd['train_labels'], dd['train_weights'] = dk.feature_pipeline.fit_transform(dd['train_features'], dd['train_labels'], dd['train_weights'])
        dd['test_features'], dd['test_labels'], dd['test_weights'] = dk.feature_pipeline.transform(dd['test_features'], dd['test_labels'], dd['test_weights'])
        dk.label_pipeline = self.define_label_pipeline(threads=dk.thread_count)
        dd['train_labels'], _, _ = dk.label_pipeline.fit_transform(dd['train_labels'])
        dd['test_labels'], _, _ = dk.label_pipeline.transform(dd['test_labels'])
        return

    def data_cleaning_predict(self, dk: FreqaiDataKitchen, pair: str) -> None:
        logger.warning(f'Your model {self.__class__.__name__} relies on the deprecated data pipeline. Please update your model to use the new data pipeline. This can be achieved by following the migration guide at {DOCS_LINK}/strategy_migration/#freqai-new-data-pipeline')
        dd = dk.data_dictionary
        dd['predict_features'], outliers, _ = dk.feature_pipeline.transform(dd['predict_features'], outlier_check=True)
        if self.freqai_info.get('DI_threshold', 0) > 0:
            dk.DI_values = dk.feature_pipeline['di'].di_values
        else:
            dk.DI_values = np.zeros(outliers.shape[0])
        dk.do_predict = outliers
        return
