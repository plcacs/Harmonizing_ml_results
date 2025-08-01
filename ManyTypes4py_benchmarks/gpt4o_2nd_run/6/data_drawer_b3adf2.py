import collections
import importlib
import logging
import re
import shutil
import threading
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import psutil
import rapidjson
from joblib.externals import cloudpickle
from numpy.typing import NDArray
from pandas import DataFrame
from freqtrade.configuration import TimeRange
from freqtrade.constants import Config
from freqtrade.data.history import load_pair_history
from freqtrade.enums import CandleType
from freqtrade.exceptions import OperationalException
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.strategy.interface import IStrategy

logger = logging.getLogger(__name__)
FEATURE_PIPELINE = 'feature_pipeline'
LABEL_PIPELINE = 'label_pipeline'
TRAINDF = 'trained_df'
METADATA = 'metadata'
METADATA_NUMBER_MODE = rapidjson.NM_NATIVE | rapidjson.NM_NAN

class pair_info(TypedDict):
    pass

class FreqaiDataDrawer:
    def __init__(self, full_path: Path, config: Config) -> None:
        self.config: Config = config
        self.freqai_info: Dict[str, Any] = config.get('freqai', {})
        self.pair_dict: Dict[str, Dict[str, Any]] = {}
        self.model_dictionary: Dict[str, Any] = {}
        self.meta_data_dictionary: Dict[str, Dict[str, Any]] = {}
        self.model_return_values: Dict[str, DataFrame] = {}
        self.historic_data: Dict[str, Dict[str, DataFrame]] = {}
        self.historic_predictions: Dict[str, DataFrame] = {}
        self.full_path: Path = full_path
        self.historic_predictions_path: Path = Path(self.full_path / 'historic_predictions.pkl')
        self.historic_predictions_bkp_path: Path = Path(self.full_path / 'historic_predictions.backup.pkl')
        self.pair_dictionary_path: Path = Path(self.full_path / 'pair_dictionary.json')
        self.global_metadata_path: Path = Path(self.full_path / 'global_metadata.json')
        self.metric_tracker_path: Path = Path(self.full_path / 'metric_tracker.json')
        self.load_drawer_from_disk()
        self.load_historic_predictions_from_disk()
        self.metric_tracker: Dict[str, Dict[str, Dict[str, List[Union[int, float]]]]] = {}
        self.load_metric_tracker_from_disk()
        self.training_queue: Dict[str, Any] = {}
        self.history_lock: threading.Lock = threading.Lock()
        self.save_lock: threading.Lock = threading.Lock()
        self.pair_dict_lock: threading.Lock = threading.Lock()
        self.metric_tracker_lock: threading.Lock = threading.Lock()
        self.old_DBSCAN_eps: Dict[str, float] = {}
        self.empty_pair_dict: Dict[str, Union[str, int, Dict[str, Any]]] = {
            'model_filename': '',
            'trained_timestamp': 0,
            'data_path': '',
            'extras': {}
        }
        self.model_type: str = self.freqai_info.get('model_save_type', 'joblib')

    def update_metric_tracker(self, metric: str, value: Union[int, float], pair: str) -> None:
        with self.metric_tracker_lock:
            if pair not in self.metric_tracker:
                self.metric_tracker[pair] = {}
            if metric not in self.metric_tracker[pair]:
                self.metric_tracker[pair][metric] = {'timestamp': [], 'value': []}
            timestamp = int(datetime.now(timezone.utc).timestamp())
            self.metric_tracker[pair][metric]['value'].append(value)
            self.metric_tracker[pair][metric]['timestamp'].append(timestamp)

    def collect_metrics(self, time_spent: float, pair: str) -> None:
        load1, load5, load15 = psutil.getloadavg()
        cpus = psutil.cpu_count()
        self.update_metric_tracker('train_time', time_spent, pair)
        self.update_metric_tracker('cpu_load1min', load1 / cpus, pair)
        self.update_metric_tracker('cpu_load5min', load5 / cpus, pair)
        self.update_metric_tracker('cpu_load15min', load15 / cpus, pair)

    def load_global_metadata_from_disk(self) -> Dict[str, Any]:
        exists = self.global_metadata_path.is_file()
        if exists:
            with self.global_metadata_path.open('r') as fp:
                metatada_dict = rapidjson.load(fp, number_mode=rapidjson.NM_NATIVE)
                return metatada_dict
        return {}

    def load_drawer_from_disk(self) -> None:
        exists = self.pair_dictionary_path.is_file()
        if exists:
            with self.pair_dictionary_path.open('r') as fp:
                self.pair_dict = rapidjson.load(fp, number_mode=rapidjson.NM_NATIVE)
        else:
            logger.info('Could not find existing datadrawer, starting from scratch')

    def load_metric_tracker_from_disk(self) -> None:
        if self.freqai_info.get('write_metrics_to_disk', False):
            exists = self.metric_tracker_path.is_file()
            if exists:
                with self.metric_tracker_path.open('r') as fp:
                    self.metric_tracker = rapidjson.load(fp, number_mode=rapidjson.NM_NATIVE)
                logger.info('Loading existing metric tracker from disk.')
            else:
                logger.info('Could not find existing metric tracker, starting from scratch')

    def load_historic_predictions_from_disk(self) -> bool:
        exists = self.historic_predictions_path.is_file()
        if exists:
            try:
                with self.historic_predictions_path.open('rb') as fp:
                    self.historic_predictions = cloudpickle.load(fp)
                logger.info(f'Found existing historic predictions at {self.full_path}, but beware that statistics may be inaccurate if the bot has been offline for an extended period of time.')
            except EOFError:
                logger.warning('Historical prediction file was corrupted. Trying to load backup file.')
                with self.historic_predictions_bkp_path.open('rb') as fp:
                    self.historic_predictions = cloudpickle.load(fp)
                logger.warning('FreqAI successfully loaded the backup historical predictions file.')
        else:
            logger.info('Could not find existing historic_predictions, starting from scratch')
        return exists

    def save_historic_predictions_to_disk(self) -> None:
        with self.historic_predictions_path.open('wb') as fp:
            cloudpickle.dump(self.historic_predictions, fp, protocol=cloudpickle.DEFAULT_PROTOCOL)
        shutil.copy(self.historic_predictions_path, self.historic_predictions_bkp_path)

    def save_metric_tracker_to_disk(self) -> None:
        with self.save_lock:
            with self.metric_tracker_path.open('w') as fp:
                rapidjson.dump(self.metric_tracker, fp, default=self.np_encoder, number_mode=rapidjson.NM_NATIVE)

    def save_drawer_to_disk(self) -> None:
        with self.save_lock:
            with self.pair_dictionary_path.open('w') as fp:
                rapidjson.dump(self.pair_dict, fp, default=self.np_encoder, number_mode=rapidjson.NM_NATIVE)

    def save_global_metadata_to_disk(self, metadata: Dict[str, Any]) -> None:
        with self.save_lock:
            with self.global_metadata_path.open('w') as fp:
                rapidjson.dump(metadata, fp, default=self.np_encoder, number_mode=rapidjson.NM_NATIVE)

    def np_encoder(self, obj: Any) -> Any:
        if isinstance(obj, np.generic):
            return obj.item()

    def get_pair_dict_info(self, pair: str) -> Tuple[str, int]:
        pair_dict = self.pair_dict.get(pair)
        if pair_dict:
            model_filename = pair_dict['model_filename']
            trained_timestamp = pair_dict['trained_timestamp']
        else:
            self.pair_dict[pair] = self.empty_pair_dict.copy()
            model_filename = ''
            trained_timestamp = 0
        return (model_filename, trained_timestamp)

    def set_pair_dict_info(self, metadata: Dict[str, Any]) -> None:
        pair_in_dict = self.pair_dict.get(metadata['pair'])
        if pair_in_dict:
            return
        else:
            self.pair_dict[metadata['pair']] = self.empty_pair_dict.copy()
            return

    def set_initial_return_values(self, pair: str, pred_df: DataFrame, dataframe: DataFrame) -> None:
        new_pred = pred_df.copy()
        new_pred['date_pred'] = dataframe['date']
        columns_to_nan = new_pred.columns.difference(['date_pred', 'date'])
        new_pred[columns_to_nan] = None
        hist_preds = self.historic_predictions[pair].copy()
        new_pred['date_pred'] = pd.to_datetime(new_pred['date_pred'])
        hist_preds['date_pred'] = pd.to_datetime(hist_preds['date_pred'])
        common_dates = pd.merge(new_pred, hist_preds, on='date_pred', how='inner')
        if len(common_dates.index) > 0:
            new_pred = new_pred.iloc[len(common_dates):]
        else:
            logger.warning(f'No common dates found between new predictions and historic predictions. You likely left your FreqAI instance offline for more than {len(dataframe.index)} candles.')
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FutureWarning)
            new_pred_reindexed = new_pred.reindex(columns=hist_preds.columns)
            df_concat = pd.concat([hist_preds, new_pred_reindexed], ignore_index=True)
        df_concat = df_concat.fillna(0)
        self.historic_predictions[pair] = df_concat
        self.model_return_values[pair] = df_concat.tail(len(dataframe.index)).reset_index(drop=True)

    def append_model_predictions(self, pair: str, predictions: DataFrame, do_preds: NDArray, dk: FreqaiDataKitchen, strat_df: DataFrame) -> None:
        len_df = len(strat_df)
        index = self.historic_predictions[pair].index[-1:]
        columns = self.historic_predictions[pair].columns
        zeros_df = pd.DataFrame(np.zeros((1, len(columns))), index=index, columns=columns)
        self.historic_predictions[pair] = pd.concat([self.historic_predictions[pair], zeros_df], ignore_index=True, axis=0)
        df = self.historic_predictions[pair]
        for label in predictions.columns:
            label_loc = df.columns.get_loc(label)
            pred_label_loc = predictions.columns.get_loc(label)
            df.iloc[-1, label_loc] = predictions.iloc[-1, pred_label_loc]
            if df[label].dtype == object:
                continue
            label_mean_loc = df.columns.get_loc(f'{label}_mean')
            label_std_loc = df.columns.get_loc(f'{label}_std')
            df.iloc[-1, label_mean_loc] = dk.data['labels_mean'][label]
            df.iloc[-1, label_std_loc] = dk.data['labels_std'][label]
        do_predict_loc = df.columns.get_loc('do_predict')
        df.iloc[-1, do_predict_loc] = do_preds[-1]
        if self.freqai_info['feature_parameters'].get('DI_threshold', 0) > 0:
            DI_values_loc = df.columns.get_loc('DI_values')
            df.iloc[-1, DI_values_loc] = dk.DI_values[-1]
        if dk.data['extra_returns_per_train']:
            rets = dk.data['extra_returns_per_train']
            for return_str in rets:
                return_loc = df.columns.get_loc(return_str)
                df.iloc[-1, return_loc] = rets[return_str]
        high_price_loc = df.columns.get_loc('high_price')
        high_loc = strat_df.columns.get_loc('high')
        df.iloc[-1, high_price_loc] = strat_df.iloc[-1, high_loc]
        low_price_loc = df.columns.get_loc('low_price')
        low_loc = strat_df.columns.get_loc('low')
        df.iloc[-1, low_price_loc] = strat_df.iloc[-1, low_loc]
        close_price_loc = df.columns.get_loc('close_price')
        close_loc = strat_df.columns.get_loc('close')
        df.iloc[-1, close_price_loc] = strat_df.iloc[-1, close_loc]
        date_pred_loc = df.columns.get_loc('date_pred')
        date_loc = strat_df.columns.get_loc('date')
        df.iloc[-1, date_pred_loc] = strat_df.iloc[-1, date_loc]
        self.model_return_values[pair] = df.tail(len_df).reset_index(drop=True)

    def attach_return_values_to_return_dataframe(self, pair: str, dataframe: DataFrame) -> DataFrame:
        df = self.model_return_values[pair]
        to_keep = [col for col in dataframe.columns if not col.startswith('&')]
        dataframe = pd.concat([dataframe[to_keep], df], axis=1)
        return dataframe

    def return_null_values_to_strategy(self, dataframe: DataFrame, dk: FreqaiDataKitchen) -> None:
        dk.find_features(dataframe)
        dk.find_labels(dataframe)
        full_labels = dk.label_list + dk.unique_class_list
        for label in full_labels:
            dataframe[label] = 0
            dataframe[f'{label}_mean'] = 0
            dataframe[f'{label}_std'] = 0
        dataframe['do_predict'] = 0
        if self.freqai_info['feature_parameters'].get('DI_threshold', 0) > 0:
            dataframe['DI_values'] = 0
        if dk.data['extra_returns_per_train']:
            rets = dk.data['extra_returns_per_train']
            for return_str in rets:
                dataframe[return_str] = 0
        dk.return_dataframe = dataframe

    def purge_old_models(self) -> None:
        num_keep = self.freqai_info['purge_old_models']
        if not num_keep:
            return
        elif isinstance(num_keep, bool):
            num_keep = 2
        model_folders = [x for x in self.full_path.iterdir() if x.is_dir()]
        pattern = re.compile('sub-train-(\\w+)_(\\d{10})')
        delete_dict: Dict[str, Dict[str, Union[int, Dict[int, Path]]]] = {}
        for directory in model_folders:
            result = pattern.match(str(directory.name))
            if result is None:
                continue
            coin = result.group(1)
            timestamp = result.group(2)
            if coin not in delete_dict:
                delete_dict[coin] = {}
                delete_dict[coin]['num_folders'] = 1
                delete_dict[coin]['timestamps'] = {int(timestamp): directory}
            else:
                delete_dict[coin]['num_folders'] += 1
                delete_dict[coin]['timestamps'][int(timestamp)] = directory
        for coin in delete_dict:
            if delete_dict[coin]['num_folders'] > num_keep:
                sorted_dict = collections.OrderedDict(sorted(delete_dict[coin]['timestamps'].items()))
                num_delete = len(sorted_dict) - num_keep
                deleted = 0
                for k, v in sorted_dict.items():
                    if deleted >= num_delete:
                        break
                    logger.info(f'Freqai purging old model file {v}')
                    shutil.rmtree(v)
                    deleted += 1

    def save_metadata(self, dk: FreqaiDataKitchen) -> None:
        if not dk.data_path.is_dir():
            dk.data_path.mkdir(parents=True, exist_ok=True)
        save_path = Path(dk.data_path)
        dk.data['data_path'] = str(dk.data_path)
        dk.data['model_filename'] = str(dk.model_filename)
        dk.data['training_features_list'] = list(dk.data_dictionary['train_features'].columns)
        dk.data['label_list'] = dk.label_list
        with (save_path / f'{dk.model_filename}_{METADATA}.json').open('w') as fp:
            rapidjson.dump(dk.data, fp, default=self.np_encoder, number_mode=METADATA_NUMBER_MODE)
        return

    def save_data(self, model: Any, coin: str, dk: FreqaiDataKitchen) -> None:
        if not dk.data_path.is_dir():
            dk.data_path.mkdir(parents=True, exist_ok=True)
        save_path = Path(dk.data_path)
        if self.model_type == 'joblib':
            with (save_path / f'{dk.model_filename}_model.joblib').open('wb') as fp:
                cloudpickle.dump(model, fp)
        elif self.model_type == 'keras':
            model.save(save_path / f'{dk.model_filename}_model.h5')
        elif self.model_type in ['stable_baselines3', 'sb3_contrib', 'pytorch']:
            model.save(save_path / f'{dk.model_filename}_model.zip')
        dk.data['data_path'] = str(dk.data_path)
        dk.data['model_filename'] = str(dk.model_filename)
        dk.data['training_features_list'] = dk.training_features_list
        dk.data['label_list'] = dk.label_list
        with (save_path / f'{dk.model_filename}_{METADATA}.json').open('w') as fp:
            rapidjson.dump(dk.data, fp, default=self.np_encoder, number_mode=METADATA_NUMBER_MODE)
        with (save_path / f'{dk.model_filename}_{FEATURE_PIPELINE}.pkl').open('wb') as fp:
            cloudpickle.dump(dk.feature_pipeline, fp)
        with (save_path / f'{dk.model_filename}_{LABEL_PIPELINE}.pkl').open('wb') as fp:
            cloudpickle.dump(dk.label_pipeline, fp)
        dk.data_dictionary['train_features'].to_pickle(save_path / f'{dk.model_filename}_{TRAINDF}.pkl')
        dk.data_dictionary['train_dates'].to_pickle(save_path / f'{dk.model_filename}_trained_dates_df.pkl')
        self.model_dictionary[coin] = model
        self.pair_dict[coin]['model_filename'] = dk.model_filename
        self.pair_dict[coin]['data_path'] = str(dk.data_path)
        if coin not in self.meta_data_dictionary:
            self.meta_data_dictionary[coin] = {}
        self.meta_data_dictionary[coin][METADATA] = dk.data
        self.meta_data_dictionary[coin][FEATURE_PIPELINE] = dk.feature_pipeline
        self.meta_data_dictionary[coin][LABEL_PIPELINE] = dk.label_pipeline
        self.save_drawer_to_disk()
        return

    def load_metadata(self, dk: FreqaiDataKitchen) -> None:
        with (dk.data_path / f'{dk.model_filename}_{METADATA}.json').open('r') as fp:
            dk.data = rapidjson.load(fp, number_mode=METADATA_NUMBER_MODE)
            dk.training_features_list = dk.data['training_features_list']
            dk.label_list = dk.data['label_list']

    def load_data(self, coin: str, dk: FreqaiDataKitchen) -> Optional[Any]:
        if not self.pair_dict[coin]['model_filename']:
            return None
        if dk.live:
            dk.model_filename = self.pair_dict[coin]['model_filename']
            dk.data_path = Path(self.pair_dict[coin]['data_path'])
        if coin in self.meta_data_dictionary:
            dk.data = self.meta_data_dictionary[coin][METADATA]
            dk.feature_pipeline = self.meta_data_dictionary[coin][FEATURE_PIPELINE]
            dk.label_pipeline = self.meta_data_dictionary[coin][LABEL_PIPELINE]
        else:
            with (dk.data_path / f'{dk.model_filename}_{METADATA}.json').open('r') as fp:
                dk.data = rapidjson.load(fp, number_mode=METADATA_NUMBER_MODE)
            with (dk.data_path / f'{dk.model_filename}_{FEATURE_PIPELINE}.pkl').open('rb') as fp:
                dk.feature_pipeline = cloudpickle.load(fp)
            with (dk.data_path / f'{dk.model_filename}_{LABEL_PIPELINE}.pkl').open('rb') as fp:
                dk.label_pipeline = cloudpickle.load(fp)
        dk.training_features_list = dk.data['training_features_list']
        dk.label_list = dk.data['label_list']
        if dk.live and coin in self.model_dictionary:
            model = self.model_dictionary[coin]
        elif self.model_type == 'joblib':
            with (dk.data_path / f'{dk.model_filename}_model.joblib').open('rb') as fp:
                model = cloudpickle.load(fp)
        elif 'stable_baselines' in self.model_type or 'sb3_contrib' == self.model_type:
            mod = importlib.import_module(self.model_type, self.freqai_info['rl_config']['model_type'])
            MODELCLASS = getattr(mod, self.freqai_info['rl_config']['model_type'])
            model = MODELCLASS.load(dk.data_path / f'{dk.model_filename}_model')
        elif self.model_type == 'pytorch':
            import torch
            zipfile = torch.load(dk.data_path / f'{dk.model_filename}_model.zip')
            model = zipfile['pytrainer']
            model = model.load_from_checkpoint(zipfile)
        if not model:
            raise OperationalException(f'Unable to load model, ensure model exists at {dk.data_path} ')
        if coin not in self.model_dictionary:
            self.model_dictionary[coin] = model
        return model

    def update_historic_data(self, strategy: IStrategy, dk: FreqaiDataKitchen) -> None:
        feat_params = self.freqai_info['feature_parameters']
        with self.history_lock:
            history_data = self.historic_data
            for pair in dk.all_pairs:
                for tf in feat_params.get('include_timeframes'):
                    hist_df = history_data[pair][tf]
                    df_dp = strategy.dp.get_pair_dataframe(pair, tf)
                    if len(df_dp.index) == 0:
                        continue
                    if str(hist_df.iloc[-1]['date']) == str(df_dp.iloc[-1:]['date'].iloc[-1]):
                        continue
                    try:
                        index = df_dp.loc[df_dp['date'] == hist_df.iloc[-1]['date']].index[0] + 1
                    except IndexError:
                        if hist_df.iloc[-1]['date'] < df_dp['date'].iloc[0]:
                            raise OperationalException(f'In memory historical data is older than oldest DataProvider candle for {pair} on timeframe {tf}')
                        else:
                            index = -1
                            logger.warning(f'No common dates in historical data and dataprovider for {pair}. Appending latest dataprovider candle to historical data but please be aware that there is likely a gap in the historical data. \nHistorical data ends at {hist_df.iloc[-1]['date']} while dataprovider starts at {df_dp['date'].iloc[0]} andends at {df_dp['date'].iloc[0]}.')
                    history_data[pair][tf] = pd.concat([hist_df, df_dp.iloc[index:]], ignore_index=True, axis=0)
            self.current_candle = history_data[dk.pair][self.config['timeframe']].iloc[-1]['date']

    def load_all_pair_histories(self, timerange: TimeRange, dk: FreqaiDataKitchen) -> None:
        history_data = self.historic_data
        for pair in dk.all_pairs:
            if pair not in history_data:
                history_data[pair] = {}
            for tf in self.freqai_info['feature_parameters'].get('include_timeframes'):
                history_data[pair][tf] = load_pair_history(
                    datadir=self.config['datadir'],
                    timeframe=tf,
                    pair=pair,
                    timerange=timerange,
                    data_format=self.config.get('dataformat_ohlcv', 'feather'),
                    candle_type=self.config.get('candle_type_def', CandleType.SPOT)
                )

    def get_base_and_corr_dataframes(self, timerange: TimeRange, pair: str, dk: FreqaiDataKitchen) -> Tuple[Dict[str, Dict[str, DataFrame]], Dict[str, DataFrame]]:
        with self.history_lock:
            corr_dataframes: Dict[str, Dict[str, DataFrame]] = {}
            base_dataframes: Dict[str, DataFrame] = {}
            historic_data = self.historic_data
            pairs = self.freqai_info['feature_parameters'].get('include_corr_pairlist', [])
            for tf in self.freqai_info['feature_parameters'].get('include_timeframes'):
                base_dataframes[tf] = dk.slice_dataframe(timerange, historic_data[pair][tf]).reset_index(drop=True)
                if pairs:
                    for p in pairs:
                        if pair in p:
                            continue
                        if p not in corr_dataframes:
                            corr_dataframes[p] = {}
                        corr_dataframes[p][tf] = dk.slice_dataframe(timerange, historic_data[p][tf]).reset_index(drop=True)
        return (corr_dataframes, base_dataframes)

    def get_timerange_from_live_historic_predictions(self) -> TimeRange:
        if not self.historic_predictions_path.is_file():
            raise OperationalException('Historic predictions not found. Historic predictions data is required to run backtest with the freqai-backtest-live-models option ')
        self.load_historic_predictions_from_disk()
        all_pairs_end_dates: List[datetime] = []
        for pair in self.historic_predictions:
            pair_historic_data = self.historic_predictions[pair]
            all_pairs_end_dates.append(pair_historic_data.date_pred.max())
        global_metadata = self.load_global_metadata_from_disk()
        start_date = datetime.fromtimestamp(int(global_metadata['start_dry_live_date']))
        end_date = max(all_pairs_end_dates)
        end_date = end_date + timedelta(days=1)
        backtesting_timerange = TimeRange('date', 'date', int(start_date.timestamp()), int(end_date.timestamp()))
        return backtesting_timerange
