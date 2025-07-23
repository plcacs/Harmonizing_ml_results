import collections
import importlib
import logging
import re
import shutil
import threading
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, TypedDict, Dict, List, Optional, Union, Tuple, cast
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

class PairInfo(TypedDict):
    model_filename: str
    trained_timestamp: int
    data_path: str
    extras: Dict[str, Any]

class FreqaiDataDrawer:
    def __init__(self, full_path: Path, config: Config) -> None:
        self.config: Config = config
        self.freqai_info: Dict[str, Any] = config.get('freqai', {})
        self.pair_dict: Dict[str, PairInfo] = {}
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
        self.empty_pair_dict: PairInfo = {'model_filename': '', 'trained_timestamp': 0, 'data_path': '', 'extras': {}}
        self.model_type: str = self.freqai_info.get('model_save_type', 'joblib')

    def update_metric_tracker(self, metric: str, value: Union[int, float], pair: str) -> None:
        with self.metric_tracker_lock:
            if pair not in self.metric_tracker:
                self.metric_tracker[pair] = {}
            if metric not in self.metric_tracker[pair]:
                self.metric_tracker[pair][metric] = {'timestamp': [], 'value': []}
            timestamp: int = int(datetime.now(timezone.utc).timestamp())
            self.metric_tracker[pair][metric]['value'].append(value)
            self.metric_tracker[pair][metric]['timestamp'].append(timestamp)

    def collect_metrics(self, time_spent: float, pair: str) -> None:
        load1, load5, load15 = psutil.getloadavg()
        cpus: int = psutil.cpu_count()
        self.update_metric_tracker('train_time', time_spent, pair)
        self.update_metric_tracker('cpu_load1min', load1 / cpus, pair)
        self.update_metric_tracker('cpu_load5min', load5 / cpus, pair)
        self.update_metric_tracker('cpu_load15min', load15 / cpus, pair)

    def load_global_metadata_from_disk(self) -> Dict[str, Any]:
        exists: bool = self.global_metadata_path.is_file()
        if exists:
            with self.global_metadata_path.open('r') as fp:
                metatada_dict: Dict[str, Any] = rapidjson.load(fp, number_mode=rapidjson.NM_NATIVE)
                return metatada_dict
        return {}

    def load_drawer_from_disk(self) -> None:
        exists: bool = self.pair_dictionary_path.is_file()
        if exists:
            with self.pair_dictionary_path.open('r') as fp:
                self.pair_dict = rapidjson.load(fp, number_mode=rapidjson.NM_NATIVE)
        else:
            logger.info('Could not find existing datadrawer, starting from scratch')

    def load_metric_tracker_from_disk(self) -> None:
        if self.freqai_info.get('write_metrics_to_disk', False):
            exists: bool = self.metric_tracker_path.is_file()
            if exists:
                with self.metric_tracker_path.open('r') as fp:
                    self.metric_tracker = rapidjson.load(fp, number_mode=rapidjson.NM_NATIVE)
                logger.info('Loading existing metric tracker from disk.')
            else:
                logger.info('Could not find existing metric tracker, starting from scratch')

    def load_historic_predictions_from_disk(self) -> bool:
        exists: bool = self.historic_predictions_path.is_file()
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
        pair_dict: Optional[PairInfo] = self.pair_dict.get(pair)
        if pair_dict:
            model_filename: str = pair_dict['model_filename']
            trained_timestamp: int = pair_dict['trained_timestamp']
        else:
            self.pair_dict[pair] = self.empty_pair_dict.copy()
            model_filename = ''
            trained_timestamp = 0
        return (model_filename, trained_timestamp)

    def set_pair_dict_info(self, metadata: Dict[str, Any]) -> None:
        pair_in_dict: Optional[PairInfo] = self.pair_dict.get(metadata['pair'])
        if pair_in_dict:
            return
        else:
            self.pair_dict[metadata['pair']] = self.empty_pair_dict.copy()
            return

    def set_initial_return_values(self, pair: str, pred_df: DataFrame, dataframe: DataFrame) -> None:
        new_pred: DataFrame = pred_df.copy()
        new_pred['date_pred'] = dataframe['date']
        columns_to_nan: pd.Index = new_pred.columns.difference(['date_pred', 'date'])
        new_pred[columns_to_nan] = None
        hist_preds: DataFrame = self.historic_predictions[pair].copy()
        new_pred['date_pred'] = pd.to_datetime(new_pred['date_pred'])
        hist_preds['date_pred'] = pd.to_datetime(hist_preds['date_pred'])
        common_dates: DataFrame = pd.merge(new_pred, hist_preds, on='date_pred', how='inner')
        if len(common_dates.index) > 0:
            new_pred = new_pred.iloc[len(common_dates):]
        else:
            logger.warning(f'No common dates found between new predictions and historic predictions. You likely left your FreqAI instance offline for more than {len(dataframe.index)} candles.')
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FutureWarning)
            new_pred_reindexed: DataFrame = new_pred.reindex(columns=hist_preds.columns)
            df_concat: DataFrame = pd.concat([hist_preds, new_pred_reindexed], ignore_index=True)
        df_concat = df_concat.fillna(0)
        self.historic_predictions[pair] = df_concat
        self.model_return_values[pair] = df_concat.tail(len(dataframe.index)).reset_index(drop=True)

    def append_model_predictions(self, pair: str, predictions: DataFrame, do_preds: NDArray[np.int_], dk: FreqaiDataKitchen, strat_df: DataFrame) -> None:
        len_df: int = len(strat_df)
        index: pd.Index = self.historic_predictions[pair].index[-1:]
        columns: pd.Index = self.historic_predictions[pair].columns
        zeros_df: DataFrame = pd.DataFrame(np.zeros((1, len(columns))), index=index, columns=columns)
        self.historic_predictions[pair] = pd.concat([self.historic_predictions[pair], zeros_df], ignore_index=True, axis=0)
        df: DataFrame = self.historic_predictions[pair]
        for label in predictions.columns:
            label_loc: int = df.columns.get_loc(label)
            pred_label_loc: int = predictions.columns.get_loc(label)
            df.iloc[-1, label_loc] = predictions.iloc[-1, pred_label_loc]
            if df[label].dtype == object:
                continue
            label_mean_loc: int = df.columns.get_loc(f'{label}_mean')
            label_std_loc: int = df.columns.get_loc(f'{label}_std')
            df.iloc[-1, label_mean_loc] = dk.data['labels_mean'][label]
            df.iloc[-1, label_std_loc] = dk.data['labels_std'][label]
        do_predict_loc: int = df.columns.get_loc('do_predict')
        df.iloc[-1, do_predict_loc] = do_preds[-1]
        if self.freqai_info['feature_parameters'].get('DI_threshold', 0) > 0:
            DI_values_loc: int = df.columns.get_loc('DI_values')
            df.iloc[-1, DI_values_loc] = dk.DI_values[-1]
        if dk.data['extra_returns_per_train']:
            rets: Dict[str, Any] = dk.data['extra_returns_per_train']
            for return_str in rets:
                return_loc: int = df.columns.get_loc(return_str)
                df.iloc[-1, return_loc] = rets[return_str]
        high_price_loc: int = df.columns.get_loc('high_price')
        high_loc: int = strat_df.columns.get_loc('high')
        df.iloc[-1, high_price_loc] = strat_df.iloc[-1, high_loc]
        low_price_loc: int = df.columns.get_loc('low_price')
        low_loc: int = strat_df.columns.get_loc('low')
        df.iloc[-1, low_price_loc] = strat_df.iloc[-1, low_loc]
        close_price_loc: int = df.columns.get_loc('close_price')
        close_loc: int = strat_df.columns.get_loc('close')
        df.iloc[-1, close_price_loc] = strat_df.iloc[-1, close_loc]
        date_pred_loc: int = df.columns.get_loc('date_pred')
        date_loc: int = strat_df.columns.get_loc('date')
        df.iloc[-1, date_pred_loc] = strat_df.iloc[-1, date_loc]
        self.model_return_values[pair] = df.tail(len_df).reset_index(drop=True)

    def attach_return_values_to_return_dataframe(self, pair: str, dataframe: DataFrame) -> DataFrame:
        df: DataFrame = self.model_return_values[pair]
        to_keep: List[str] = [col for col in dataframe.columns if not col.startswith('&')]
        dataframe = pd.concat([dataframe[to_keep], df], axis=1)
        return dataframe

    def return_null_values_to_strategy(self, dataframe: DataFrame, dk: FreqaiDataKitchen) -> None:
        dk.find_features(dataframe)
        dk.find_labels(dataframe)
        full_labels: List[str] = dk.label_list + dk.unique_class_list
        for label in full_labels:
            dataframe[label] = 0
            dataframe[f'{label}_mean'] = 0
            dataframe[f'{label}_std'] = 0
        dataframe['do_predict'] = 0
        if self.freqai_info['feature_parameters'].get('DI_threshold', 0) > 0:
            dataframe['DI_values'] = 0
        if dk.data['extra_returns_per_train']:
            rets: Dict[str, Any] = dk.data['extra_returns_per_train']
            for return_str in rets:
                dataframe[return_str] = 0
        dk.return_dataframe = dataframe

    def purge_old_models(self) -> None:
        num_keep: Union[bool, int] = self.freqai_info['purge_old_models']
        if not num_keep:
            return
        elif isinstance(num_keep, bool):
            num_keep = 2
        model_folders: List[Path] = [x for x in self.full_path.iterdir() if x.is_dir()]
        pattern: re.Pattern = re.compile('sub-train-(\\w+)_(\\d{10})')
        delete_dict: Dict[str, Dict[str, Union[int, Dict[int, Path]]]] = {}
        for directory in model_folders:
            result: Optional[re.Match] = pattern.match(str(directory.name))
            if result is None:
                continue
            coin: str = result.group(1)
            timestamp: str = result.group(2)
            if coin not in delete_dict:
                delete_dict[coin] = {}
                delete_dict[coin]['num_folders'] = 1
                delete_dict[coin]['timestamps'] = {int(timestamp): directory}
            else:
                delete_dict[coin]['num_folders'] += 1
                delete_dict[coin]['timestamps'][int(timestamp)] = directory
        for coin in delete_dict:
            if delete_dict[coin]['num_folders'] > num_keep:
                sorted_dict: collections.OrderedDict = collections.OrderedDict(sorted(delete_dict[coin]['timestamps'].items()))
                num_delete: int = len(sorted_dict) - num_keep
                deleted: int = 0
                for k, v in sorted_dict.items():
                    if deleted >= num_delete:
                        break
                    logger.info(f'Freqai purging old model file {v}')
                    shutil.rmtree(v)
                    deleted += 1

    def save_metadata(self, dk: FreqaiDataKitchen) -> None:
        if not dk.data_path.is_dir():
            dk.data_path.mkdir(parents=True, exist_ok=True)
        save_path: Path = Path(dk.data_path)
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
        save_path: Path = Path(dk.data_path)
        if self.model_type == 'joblib':
            with (save_path / f'{dk.model_filename}_model.joblib').open('wb') as fp:
                cloudpickle.dump(model, fp