import collections
import importlib
import logging
import re
import shutil
import threading
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, TypedDict, Optional, Dict, List, Tuple, Union
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
    model_filename: str
    trained_timestamp: int
    data_path: str
    extras: dict

class FreqaiDataDrawer:
    """
    Class aimed at holding all pair models/info in memory for better inferencing/retrainig/saving
    /loading to/from disk.
    This object remains persistent throughout live/dry.

    Record of contribution:
    FreqAI was developed by a group of individuals who all contributed specific skillsets to the
    project.

    Conception and software development:
    Robert Caulk @robcaulk

    Theoretical brainstorming:
    Elin Törnquist @th0rntwig

    Code review, software architecture brainstorming:
    @xmatthias

    Beta testing and bug reporting:
    @bloodhunter4rc, Salah Lamkadem @ikonx, @ken11o2, @longyu, @paranoidandy, @smidelis, @smarm
    Juha Nykänen @suikula, Wagner Costa @wagnercosta, Johan Vlugt @Jooopieeert
    """

    def __init__(self, full_path: Path, config: Config) -> None:
        self.config: Config = config
        self.freqai_info: dict = config.get('freqai', {})
        self.pair_dict: Dict[str, pair_info] = {}
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
        self.metric_tracker: Dict[str, Dict[str, Dict[str, list]]] = {}
        self.load_metric_tracker_from_disk()
        self.training_queue: Dict[str, int] = {}
        self.history_lock: threading.Lock = threading.Lock()
        self.save_lock: threading.Lock = threading.Lock()
        self.pair_dict_lock: threading.Lock = threading.Lock()
        self.metric_tracker_lock: threading.Lock = threading.Lock()
        self.old_DBSCAN_eps: Dict[str, float] = {}
        self.empty_pair_dict: pair_info = {'model_filename': '', 'trained_timestamp': 0, 'data_path': '', 'extras': {}}
        self.model_type: str = self.freqai_info.get('model_save_type', 'joblib')
        self.current_candle: Optional[Any] = None

    def update_metric_tracker(self, metric: str, value: Any, pair: str) -> None:
        """
        General utility for adding and updating custom metrics. Typically used
        for adding training performance, train timings, inferenc timings, cpu loads etc.
        """
        with self.metric_tracker_lock:
            if pair not in self.metric_tracker:
                self.metric_tracker[pair] = {}
            if metric not in self.metric_tracker[pair]:
                self.metric_tracker[pair][metric] = {'timestamp': [], 'value': []}
            timestamp: int = int(datetime.now(timezone.utc).timestamp())
            self.metric_tracker[pair][metric]['value'].append(value)
            self.metric_tracker[pair][metric]['timestamp'].append(timestamp)

    def collect_metrics(self, time_spent: float, pair: str) -> None:
        """
        Add metrics to the metric tracker dictionary
        """
        (load1, load5, load15) = psutil.getloadavg()
        cpus: int = psutil.cpu_count()
        self.update_metric_tracker('train_time', time_spent, pair)
        self.update_metric_tracker('cpu_load1min', load1 / cpus, pair)
        self.update_metric_tracker('cpu_load5min', load5 / cpus, pair)
        self.update_metric_tracker('cpu_load15min', load15 / cpus, pair)

    def load_global_metadata_from_disk(self) -> dict:
        """
        Locate and load a previously saved global metadata in present model folder.
        """
        exists: bool = self.global_metadata_path.is_file()
        if exists:
            with self.global_metadata_path.open('r') as fp:
                metatada_dict: dict = rapidjson.load(fp, number_mode=rapidjson.NM_NATIVE)
                return metatada_dict
        return {}

    def load_drawer_from_disk(self) -> None:
        """
        Locate and load a previously saved data drawer full of all pair model metadata in
        present model folder.
        Load any existing metric tracker that may be present.
        """
        exists: bool = self.pair_dictionary_path.is_file()
        if exists:
            with self.pair_dictionary_path.open('r') as fp:
                self.pair_dict = rapidjson.load(fp, number_mode=rapidjson.NM_NATIVE)
        else:
            logger.info('Could not find existing datadrawer, starting from scratch')

    def load_metric_tracker_from_disk(self) -> None:
        """
        Tries to load an existing metrics dictionary if the user
        wants to collect metrics.
        """
        if self.freqai_info.get('write_metrics_to_disk', False):
            exists: bool = self.metric_tracker_path.is_file()
            if exists:
                with self.metric_tracker_path.open('r') as fp:
                    self.metric_tracker = rapidjson.load(fp, number_mode=rapidjson.NM_NATIVE)
                logger.info('Loading existing metric tracker from disk.')
            else:
                logger.info('Could not find existing metric tracker, starting from scratch')

    def load_historic_predictions_from_disk(self) -> bool:
        """
        Locate and load a previously saved historic predictions.
        :return: bool - whether or not the drawer was located
        """
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
        """
        Save historic predictions pickle to disk
        """
        with self.historic_predictions_path.open('wb') as fp:
            cloudpickle.dump(self.historic_predictions, fp, protocol=cloudpickle.DEFAULT_PROTOCOL)
        shutil.copy(self.historic_predictions_path, self.historic_predictions_bkp_path)

    def save_metric_tracker_to_disk(self) -> None:
        """
        Save metric tracker of all pair metrics collected.
        """
        with self.save_lock:
            with self.metric_tracker_path.open('w') as fp:
                rapidjson.dump(self.metric_tracker, fp, default=self.np_encoder, number_mode=rapidjson.NM_NATIVE)

    def save_drawer_to_disk(self) -> None:
        """
        Save data drawer full of all pair model metadata in present model folder.
        """
        with self.save_lock:
            with self.pair_dictionary_path.open('w') as fp:
                rapidjson.dump(self.pair_dict, fp, default=self.np_encoder, number_mode=rapidjson.NM_NATIVE)

    def save_global_metadata_to_disk(self, metadata: dict) -> None:
        """
        Save global metadata json to disk
        """
        with self.save_lock:
            with self.global_metadata_path.open('w') as fp:
                rapidjson.dump(metadata, fp, default=self.np_encoder, number_mode=rapidjson.NM_NATIVE)

    def np_encoder(self, obj: Any) -> Any:
        if isinstance(obj, np.generic):
            return obj.item()

    def get_pair_dict_info(self, pair: str) -> Tuple[str, int]:
        """
        Locate and load existing model metadata from persistent storage. If not located,
        create a new one and append the current pair to it and prepare it for its first
        training
        :param pair: str: pair to lookup
        :return:
            model_filename: str = unique filename used for loading persistent objects from disk
            trained_timestamp: int = the last time the coin was trained
        """
        pair_dict: Optional[pair_info] = self.pair_dict.get(pair)
        if pair_dict:
            model_filename: str = pair_dict['model_filename']
            trained_timestamp: int = pair_dict['trained_timestamp']
        else:
            self.pair_dict[pair] = self.empty_pair_dict.copy()
            model_filename = ''
            trained_timestamp = 0
        return (model_filename, trained_timestamp)

    def set_pair_dict_info(self, metadata: dict) -> None:
        pair_in_dict: Optional[pair_info] = self.pair_dict.get(metadata['pair'])
        if pair_in_dict:
            return
        else:
            self.pair_dict[metadata['pair']] = self.empty_pair_dict.copy()
            return

    def set_initial_return_values(self, pair: str, pred_df: DataFrame, dataframe: DataFrame) -> None:
        """
        Set the initial return values to the historical predictions dataframe. This avoids needing
        to repredict on historical candles, and also stores historical predictions despite
        retrainings (so stored predictions are true predictions, not just inferencing on trained
        data).

        We also aim to keep the date from historical predictions so that the FreqUI displays
        zeros during any downtime (between FreqAI reloads).
        """
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
            df_concat: DataFrame = pd.concat([hist_preds, new_pred_reindexed], ignore_index=True, axis=0)
        df_concat = df_concat.fillna(0)
        self.historic_predictions[pair] = df_concat
        self.model_return_values[pair] = df_concat.tail(len(dataframe.index)).reset_index(drop=True)

    def append_model_predictions(self, pair: str, predictions: DataFrame, do_preds: NDArray[np.int_], dk: FreqaiDataKitchen, strat_df: DataFrame) -> None:
        """
        Append model predictions to historic predictions dataframe, then set the
        strategy return dataframe to the tail of the historic predictions. The length of
        the tail is equivalent to the length of the dataframe that entered FreqAI from
        the strategy originally. Doing this allows FreqUI to always display the correct
        historic predictions.
        """
        len_df: int = len(strat_df)
        index: pd.RangeIndex = self.historic_predictions[pair].index[-1:]
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
            rets: dict = dk.data['extra_returns_per_train']
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
        """
        Attach the return values to the strat dataframe
        :param dataframe: DataFrame = strategy dataframe
        :return: DataFrame = strat dataframe with return values attached
        """
        df: DataFrame = self.model_return_values[pair]
        to_keep: List[str] = [col for col in dataframe.columns if not col.startswith('&')]
        dataframe = pd.concat([dataframe[to_keep], df], axis=1)
        return dataframe

    def return_null_values_to_strategy(self, dataframe: DataFrame, dk: FreqaiDataKitchen) -> None:
        """
        Build 0 filled dataframe to return to strategy
        """
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
            rets: dict = dk.data['extra_returns_per_train']
            for return_str in rets:
                dataframe[return_str] = 0
        dk.return_dataframe = dataframe

