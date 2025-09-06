import copy
import inspect
import logging
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import numpy.typing as npt
import pandas as pd
import psutil
from datasieve.pipeline import Pipeline
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from freqtrade.configuration import TimeRange
from freqtrade.constants import DOCS_LINK, Config
from freqtrade.data.converter import reduce_dataframe_footprint
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import timeframe_to_seconds
from freqtrade.strategy import merge_informative_pair
from freqtrade.strategy.interface import IStrategy
pd.set_option('future.no_silent_downcasting', True)
SECONDS_IN_DAY: int = 86400
SECONDS_IN_HOUR: int = 3600
logger: logging.Logger = logging.getLogger(__name__)


class FreqaiDataKitchen:
    def __init__(self, config: dict, live: bool = False, pair: str = ''):
        self.data: dict = {}
        self.data_dictionary: dict = {}
        self.config: dict = config
        self.freqai_config: dict = config['freqai']
        self.full_df: DataFrame = DataFrame()
        self.append_df: DataFrame = DataFrame()
        self.data_path: Path = Path()
        self.label_list: list = []
        self.training_features_list: list = []
        self.model_filename: str = ''
        self.backtesting_results_path: Path = Path()
        self.backtest_predictions_folder: str = 'backtesting_predictions'
        self.live: bool = live
        self.pair: str = pair
        self.keras: bool = self.freqai_config.get('keras', False)
        self.set_all_pairs()
        self.backtest_live_models: bool = config.get('freqai_backtest_live_models', False)
        self.feature_pipeline: Pipeline = Pipeline()
        self.label_pipeline: Pipeline = Pipeline()
        self.DI_values: np.ndarray = np.array([])
        if not self.live:
            self.full_path: Path = self.get_full_models_path(self.config)
            if not self.backtest_live_models:
                self.full_timerange, self.training_timeranges, self.backtesting_timeranges = self.split_timerange(self.full_timerange, config['freqai']['train_period_days'], config['freqai']['backtest_period_days'])
        self.data['extra_returns_per_train']: dict = self.freqai_config.get('extra_returns_per_train', {})
        if not self.freqai_config.get('data_kitchen_thread_count', 0):
            self.thread_count: int = max(int(psutil.cpu_count() * 2 - 2), 1)
        else:
            self.thread_count: int = self.freqai_config['data_kitchen_thread_count']
        self.train_dates: DataFrame = pd.DataFrame()
        self.unique_classes: dict = {}
        self.unique_class_list: list = []
        self.backtest_live_models_data: dict = {}

    def func_zsmu6cv0(self, pair: str, trained_timestamp: int = None) -> None:
        self.full_path: Path = self.get_full_models_path(self.config)
        self.data_path: Path = Path(self.full_path / f'sub-train-{pair.split()[0]}_{trained_timestamp}')
        return

    def func_4k5yrwho(self, filtered_dataframe: DataFrame, labels: DataFrame) -> dict:
        feat_dict: dict = self.freqai_config['feature_parameters']
        if 'shuffle' not in self.freqai_config['data_split_parameters']:
            self.freqai_config['data_split_parameters'].update({'shuffle': False})
        if feat_dict.get('weight_factor', 0) > 0:
            weights: np.ndarray = self.set_weights_higher_recent(len(filtered_dataframe))
        else:
            weights: np.ndarray = np.ones(len(filtered_dataframe))
        if self.freqai_config.get('data_split_parameters', {}).get('test_size', 0.1) != 0:
            train_features, test_features, train_labels, test_labels, train_weights, test_weights = train_test_split(filtered_dataframe[:filtered_dataframe.shape[0], labels, weights, **self.config['freqai']['data_split_parameters'])
        else:
            test_labels: np.ndarray = np.zeros(2)
            test_features: DataFrame = pd.DataFrame()
            test_weights: np.ndarray = np.zeros(2)
            train_features: DataFrame = filtered_dataframe
            train_labels: DataFrame = labels
            train_weights: np.ndarray = weights
        if feat_dict['shuffle_after_split']:
            rint1: int = random.randint(0, 100)
            rint2: int = random.randint(0, 100)
            train_features: DataFrame = train_features.sample(frac=1, random_state=rint1).reset_index(drop=True)
            train_labels: DataFrame = train_labels.sample(frac=1, random_state=rint1).reset_index(drop=True)
            train_weights: np.ndarray = pd.DataFrame(train_weights).sample(frac=1, random_state=rint1).reset_index(drop=True).to_numpy()[:, 0]
            test_features: DataFrame = test_features.sample(frac=1, random_state=rint2).reset_index(drop=True)
            test_labels: DataFrame = test_labels.sample(frac=1, random_state=rint2).reset_index(drop=True)
            test_weights: np.ndarray = pd.DataFrame(test_weights).sample(frac=1, random_state=rint2).reset_index(drop=True).to_numpy()[:, 0]
        if self.freqai_config['feature_parameters'].get('reverse_train_test_order', False):
            return self.build_data_dictionary(test_features, train_features, test_labels, train_labels, test_weights, train_weights)
        else:
            return self.build_data_dictionary(train_features, test_features, train_labels, test_labels, train_weights, test_weights)

    def func_laxfowo1(self, unfiltered_df: DataFrame, training_feature_list: list, label_list: list = [], training_filter: bool = True) -> tuple:
        filtered_df: DataFrame = unfiltered_df.filter(training_feature_list, axis=1)
        filtered_df: DataFrame = filtered_df.replace([np.inf, -np.inf], np.nan)
        drop_index: pd.Series = pd.isnull(filtered_df).any(axis=1)
        drop_index: pd.Series = drop_index.replace(True, 1).replace(False, 0).infer_objects(copy=False)
        if training_filter:
            labels: DataFrame = unfiltered_df.filter(label_list, axis=1)
            drop_index_labels: pd.Series = pd.isnull(labels).any(axis=1)
            drop_index_labels: pd.Series = drop_index_labels.replace(True, 1).replace(False, 0).infer_objects(copy=False)
            dates: pd.Series = unfiltered_df['date']
            filtered_df: DataFrame = filtered_df[(drop_index == 0) & (drop_index_labels == 0)]
            labels: DataFrame = labels[(drop_index == 0) & (drop_index_labels == 0)]
            self.train_dates: pd.Series = dates[(drop_index == 0) & (drop_index_labels == 0)]
            logger.info(f'{self.pair}: dropped {len(unfiltered_df) - len(filtered_df)} training points due to NaNs in populated dataset {len(unfiltered_df)}.')
            if len(filtered_df) == 0 and not self.live:
                raise OperationalException(f"{self.pair}: all training data dropped due to NaNs. You likely did not download enough training data prior to your backtest timerange. Hint: {DOCS_LINK}/freqai-running/#downloading-data-to-cover-the-full-backtest-period")
            if 1 - len(filtered_df) / len(unfiltered_df) > 0.1 and self.live:
                worst_indicator: str = str(unfiltered_df.count().idxmin())
                logger.warning(f' {(1 - len(filtered_df) / len(unfiltered_df)) * 100:.0f} percent  of training data dropped due to NaNs, model may perform inconsistent with expectations. Verify {worst_indicator}')
            self.data['filter_drop_index_training']: pd.Series = drop_index
        else:
            drop_index: pd.Series = pd.isnull(filtered_df).any(axis=1)
            self.data['filter_drop_index_prediction']: pd.Series = drop_index
            filtered_df.fillna(0, inplace=True)
            drop_index: np.ndarray = ~drop_index
            self.do_predict: np.ndarray = np.array(drop_index.replace(True, 1).replace(False, 0))
            if len(self.do_predict) - self.do_predict.sum() > 0:
                logger.info(f'dropped {len(self.do_predict) - self.do_predict.sum()} of {len(filtered_df)} prediction data points due to NaNs.')
            labels: list = []
        return filtered_df, labels

    def func_nrz1zx3m(self, train_df: DataFrame, test_df: DataFrame, train_labels: DataFrame, test_labels: DataFrame, train_weights: np.ndarray, test_weights: np.ndarray) -> dict:
        self.data_dictionary: dict = {'train_features': train_df, 'test_features': test_df, 'train_labels': train_labels, 'test_labels': test_labels, 'train_weights': train_weights, 'test_weights': test_weights, 'train_dates': self.train_dates}
        return self.data_dictionary

    def func_5au0tulw(self, tr: str, train_split: int = 28, bt_split: int = 7) -> tuple:
        if not isinstance(train_split, int) or train_split < 1:
            raise OperationalException(f'train_period_days must be an integer greater than 0. Got {train_split}.')
        train_period_days: int = train_split * SECONDS_IN_DAY
        bt_period: int = bt_split * SECONDS_IN_DAY
        full_timerange: TimeRange = TimeRange.parse_timerange(tr)
        config_timerange: TimeRange = TimeRange.parse_timerange(self.config['timerange'])
        if config_timerange.stopts == 0:
            config_timerange.stopts = int(datetime.now(tz=timezone.utc).timestamp())
        timerange_train: TimeRange = copy.deepcopy(full_timerange)
        timerange_backtest: TimeRange = copy.deepcopy(full_timerange)
        tr_training_list: list = []
        tr_backtesting_list: list = []
        tr_training_list_timerange: list = []
        tr_backtesting_list_timerange: list = []
        first: bool = True
        while True:
            if not first:
                timerange_train.startts += int(bt_period)
            timerange_train.stopts = (timerange_train.startts + train_period_days)
            first = False
            tr_training_list.append(timerange_train.timerange_str)
            tr_training_list_timerange.append(copy.deepcopy(timerange_train))
            timerange_backtest.startts = timerange_train.stopts
            timerange_backtest.stopts = timerange_backtest.startts + int(bt_period)
            if timerange_backtest.stopts > config_timerange.stopts:
                timerange_backtest.stopts = config_timerange.stopts
            tr_backtesting_list.append(timerange_backtest.timerange_str)
            tr_backtesting_list_timerange.append(copy.deepcopy(timerange_backtest))
            if timerange_backtest.stopts == config_timerange.stopts:
                break
        return tr_training_list_timerange, tr_backtesting_list_timerange

    def func_arhql2n5(self, timerange: TimeRange, df: DataFrame) -> DataFrame:
        if not self.live:
            df: DataFrame = df.loc[(df['date'] >= timerange.startdt) & (df['date'] < timerange.stopdt), :]
        else:
            df: DataFrame = df.loc[df['date'] >= timerange.startdt, :]
        return df

    def func_9cevie8g(self, dataframe: DataFrame) -> None:
        column_names: list = dataframe.columns
        features: list = [c for c in column_names if '%' in c]
        if not features:
            raise OperationalException('Could not find any features!')
        self.training_features_list: list = features

    def func_w6rclape(self, dataframe: DataFrame) -> None:
        column_names: list = dataframe.columns
        labels: list = [c for c in column_names if '&' in c]
        self.label_list: list = labels

    def func_ttullx9p(self, num_weights: int) -> np.ndarray:
        wfactor: int = self.config['freqai']['feature_parameters']['weight_factor']
        weights: np.ndarray = np.exp(-np.arange(num_weights) / (wfactor * num_weights))[::-1]
        return weights

    def func_nfjs0ia2(self, predictions: DataFrame, do_predict: np.ndarray, dataframe_backtest: DataFrame) -> DataFrame:
        append_df: DataFrame = DataFrame()
        for label in predictions.columns:
            append_df[label]: pd.Series = predictions[label]
            if append_df[label].dtype == object:
                continue
            if 'labels_mean' in self.data:
                append_df[f'{label}_mean']: pd.Series = self.data['labels_mean'][label]
            if 'labels_std' in self.data:
                append_df[f'{label}_std']: pd.Series = self.data['labels_std'][label]
        for extra_col in self.data['extra_returns_per_train']:
            append_df[f'{extra_col}']: pd.Series = self.data['extra_returns_per_train'][extra_col]
        append_df['do_predict']: np.ndarray = do_predict
        if self.freqai_config['feature_parameters'].get('DI_threshold', 0) > 0:
            append_df['DI_values']: np.ndarray = self.DI_values
        user_cols: list = [col for col in dataframe_backtest.columns if col.startswith('%%')]
        cols: list = ['date']
        cols.extend(user_cols)
        dataframe_backtest.reset_index(drop=True, inplace=True)
        merged_df: DataFrame = pd.concat([dataframe_backtest[cols], append_df], axis=1)
        return merged_df

    def func_uz7nyiii(self, append_df: DataFrame) -> None:
        if self.full_df.empty:
            self.full_df: DataFrame = append_df
        else:
            self.full_df: DataFrame = pd.concat([self.full_df, append_df], axis=0, ignore_index=True)

    def func_ifge3wnf(self, dataframe: DataFrame) -> None:
        self.find_labels(dataframe)
        for key in self.label_list:
            if dataframe[key].dtype == object:
                self.unique_classes[key]: np.ndarray = dataframe[key].dropna().unique()
        if self.unique_classes:
            for label in self.unique_classes:
                self.unique_class_list += list(self.unique_classes[label])

    def func_alf38v0c(self, dataframe: DataFrame) -> dict:
        corr_dataframes: dict = {}
        pairs: list = self.freqai_config['feature_parameters'].get('include_corr_pairlist', [])
        for pair in pairs:
            pair: str = pair.replace(':', '')
            pair_cols: list = [col for col in dataframe.columns if col.startswith('%') and f'{pair}_' in col]
            if pair_cols:
                pair_cols.insert(0, 'date')
                corr_dataframes[pair]: DataFrame = dataframe.filter(pair_cols, axis=1)
        return corr_dataframes

    def func_psq937a0(self, dataframe: DataFrame, corr_dataframes: dict, current_pair: str) -> DataFrame:
        pairs: list = self.freqai_config['feature_parameters'].get('include_corr_pairlist', [])
        current_pair: str = current_pair.replace(':', '')
        for pair in pairs:
            pair: str = pair.replace(':', '')
            if current_pair != pair:
                dataframe: DataFrame = dataframe.merge(corr_dataframes[pair], how='left', on='date')
        return dataframe

    def func_ufxbr0cl(self, pair: str, tf: str, strategy: IStrategy, corr_dataframes: dict = {}, base_dataframes: dict = {}, is_corr_pairs: bool = False) -> DataFrame:
        if is_corr_pairs:
            dataframe: DataFrame = corr_dataframes[pair][tf]
            if not dataframe.empty:
                return dataframe
            else:
                dataframe: DataFrame = strategy.dp.get_pair_dataframe(pair=pair, timeframe=tf)
                return dataframe
        else:
            dataframe: DataFrame = base_dataframes[tf]
            if not dataframe.empty:
                return dataframe
            else:
                dataframe: DataFrame = strategy.dp.get_pair_dataframe(pair=pair, timeframe=tf)
                return dataframe

    def func_v9k00qzd(self, df_main: DataFrame, df_to_merge: DataFrame, tf: str, timeframe_inf: str, suffix: str) -> DataFrame:
        dataframe: DataFrame = merge_informative_pair(df_main, df_to_merge, tf, timeframe_inf=timeframe_inf, append_timeframe=False, suffix=suffix, ffill=True)
        skip_columns: list = ['{s}_{suffix}' for s in ['date', 'open', 'high', 'low', 'close', 'volume']]
        dataframe: DataFrame = dataframe.drop(columns=skip_columns)
        return dataframe

    def func_30tumvq2(self, dataframe: DataFrame, pair: str, strategy: IStrategy, corr_dataframes: dict, base_dataframes: dict, is_corr_pairs: bool = False) -> DataFrame:
        tfs: list = self.freqai_config['feature_parameters'].get('include_timeframes')
        for tf in tfs:
            metadata: dict = {'pair': pair, 't': tf}
            informative_df: DataFrame = self.get_pair_data_for_features(pair, tf, strategy, corr_dataframes, base_dataframes, is_corr_pairs)
            informative_copy: DataFrame = informative_df.copy()
            logger.debug(f'Populating features for {pair} {tf}')
            for t in self.freqai_config['feature_parameters']['indicator_periods_candles']:
                df_features: DataFrame = strategy.feature_engineering_expand_all(informative_copy.copy(), t, metadata=metadata)
                suffix: str = f'{t}'
                informative_df: DataFrame = self.merge_features(informative_df, df_features, tf, tf, suffix)
            generic_df: DataFrame = strategy.feature_engineering_expand_basic(informative_copy.copy(), metadata=metadata)
            suffix: str = 'gen'
            informative_df: DataFrame = self.merge_features(informative_df, generic_df, tf, tf, suffix)
            indicators: list = [col for col in informative_df if col.startswith('%')]
            for n in range(self.freqai_config['feature_parameters']['include_shifted_candles'] + 1):
                if n == 0:
                    continue
                df_shift: DataFrame = informative_df[indicators].shift(n)
                df_shift: DataFrame = df_shift.add_suffix('_shift-' + str(n))
                informative_df: DataFrame = pd.concat((informative_df, df_shift), axis=1)
            dataframe: DataFrame = self.merge_features(dataframe.copy(), informative_df, self.config['timeframe'], tf, f'{pair}_{tf}')
        return dataframe

    def func_j5xkv503(self, strategy: IStrategy, corr_dataframes: dict = {}, base_dataframes: dict = {}, pair: str = '', prediction_dataframe: DataFrame = pd.DataFrame(), do_corr_pairs: bool = True) -> DataFrame:
        new_version: bool = inspect.getsource(strategy.populate_any_indicators) == inspect.getsource(IStrategy.populate_any_indicators)
       