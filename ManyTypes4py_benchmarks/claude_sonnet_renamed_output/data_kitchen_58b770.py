import copy
import inspect
import logging
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
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
SECONDS_IN_DAY = 86400
SECONDS_IN_HOUR = 3600
logger = logging.getLogger(__name__)


class FreqaiDataKitchen:
    """
    Class designed to analyze data for a single pair. Employed by the IFreqaiModel class.
    Functionalities include holding, saving, loading, and analyzing the data.

    This object is not persistent, it is reinstantiated for each coin, each time the coin
    model needs to be inferenced or trained.

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

    def __init__(self, config: Dict[str, Any], live: bool = False, pair: str = '') -> None:
        self.data: Dict[str, Any] = {}
        self.data_dictionary: Dict[str, Any] = {}
        self.config: Dict[str, Any] = config
        self.freqai_config: Dict[str, Any] = config['freqai']
        self.full_df: DataFrame = DataFrame()
        self.append_df: DataFrame = DataFrame()
        self.data_path: Path = Path()
        self.label_list: List[str] = []
        self.training_features_list: List[str] = []
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
        self.DI_values: npt.NDArray[Any] = np.array([])
        if not self.live:
            self.full_path: Path = self.get_full_models_path(self.config)
            if not self.backtest_live_models:
                self.full_timerange: TimeRange = self.create_fulltimerange(
                    self.config['timerange'], 
                    self.freqai_config.get('train_period_days', 0)
                )
                self.training_timeranges: List[TimeRange]
                self.backtesting_timeranges: List[TimeRange]
                (
                    self.training_timeranges,
                    self.backtesting_timeranges
                ) = self.split_timerange(
                    self.full_timerange, 
                    config['freqai']['train_period_days'], 
                    config['freqai']['backtest_period_days']
                )
        self.data['extra_returns_per_train'] = self.freqai_config.get('extra_returns_per_train', {})
        if not self.freqai_config.get('data_kitchen_thread_count', 0):
            self.thread_count: int = max(int(psutil.cpu_count() * 2 - 2), 1)
        else:
            self.thread_count: int = self.freqai_config['data_kitchen_thread_count']
        self.train_dates: DataFrame = pd.DataFrame()
        self.unique_classes: Dict[str, Any] = {}
        self.unique_class_list: List[Any] = []
        self.backtest_live_models_data: Dict[str, Any] = {}

    def func_zsmu6cv0(
        self, 
        pair: str, 
        trained_timestamp: Optional[int] = None
    ) -> None:
        """
        Set the paths to the data for the present coin/botloop
        :param pair: str = trading pair
        :param trained_timestamp: Optional[int] = timestamp of most recent training
        """
        self.full_path = self.get_full_models_path(self.config)
        self.data_path = Path(self.full_path / f"sub-train-{pair.split('/')[0]}_{trained_timestamp}")
        return

    def func_4k5yrwho(
        self, 
        filtered_dataframe: DataFrame, 
        labels: DataFrame
    ) -> Dict[str, Any]:
        """
        Given the dataframe for the full history for training, split the data into
        training and test data according to user specified parameters in configuration
        file.
        :param filtered_dataframe: DataFrame = cleaned dataframe ready to be split.
        :param labels: DataFrame = cleaned labels ready to be split.
        :return: Dict[str, Any] = data dictionary containing train and test splits
        """
        feat_dict: Dict[str, Any] = self.freqai_config['feature_parameters']
        if 'shuffle' not in self.freqai_config['data_split_parameters']:
            self.freqai_config['data_split_parameters'].update({'shuffle': False})
        if feat_dict.get('weight_factor', 0) > 0:
            weights: npt.NDArray[Any] = self.set_weights_higher_recent(len(filtered_dataframe))
        else:
            weights = np.ones(len(filtered_dataframe))
        if self.freqai_config.get('data_split_parameters', {}).get('test_size', 0.1) != 0:
            train_features: DataFrame
            test_features: DataFrame
            train_labels: DataFrame
            test_labels: DataFrame
            train_weights: npt.NDArray[Any]
            test_weights: npt.NDArray[Any]
            (
                train_features, 
                test_features, 
                train_labels, 
                test_labels,
                train_weights, 
                test_weights
            ) = train_test_split(
                filtered_dataframe[:filtered_dataframe.shape[0]], 
                labels,
                weights, 
                **self.config['freqai']['data_split_parameters']
            )
        else:
            test_labels: np.ndarray = np.zeros(2)
            test_features: DataFrame = pd.DataFrame()
            test_weights: np.ndarray = np.zeros(2)
            train_features: DataFrame = filtered_dataframe
            train_labels: DataFrame = labels
            train_weights: npt.NDArray[Any] = weights
        if feat_dict['shuffle_after_split']:
            rint1: int = random.randint(0, 100)
            rint2: int = random.randint(0, 100)
            train_features = train_features.sample(frac=1, random_state=rint1).reset_index(drop=True)
            train_labels = train_labels.sample(frac=1, random_state=rint1).reset_index(drop=True)
            train_weights = pd.DataFrame(train_weights).sample(frac=1, random_state=rint1).reset_index(drop=True).to_numpy()[:, 0]
            test_features = test_features.sample(frac=1, random_state=rint2).reset_index(drop=True)
            test_labels = test_labels.sample(frac=1, random_state=rint2).reset_index(drop=True)
            test_weights = pd.DataFrame(test_weights).sample(frac=1, random_state=rint2).reset_index(drop=True).to_numpy()[:, 0]
        if self.freqai_config['feature_parameters'].get('reverse_train_test_order', False):
            return self.build_data_dictionary(
                test_features, 
                train_features,
                test_labels, 
                train_labels, 
                test_weights, 
                train_weights
            )
        else:
            return self.build_data_dictionary(
                train_features, 
                test_features,
                train_labels, 
                test_labels, 
                train_weights, 
                test_weights
            )

    def func_laxfowo1(
        self, 
        unfiltered_df: DataFrame, 
        training_feature_list: List[str],
        label_list: Optional[List[str]] = None, 
        training_filter: bool = True
    ) -> Tuple[DataFrame, Union[DataFrame, List[Any]]]:
        """
        Filter the unfiltered dataframe to extract the user requested features/labels and properly
        remove all NaNs. Any row with a NaN is removed from training dataset or replaced with
        0s in the prediction dataset. However, prediction dataset do_predict will reflect any
        row that had a NaN and will shield user from that prediction.

        :param unfiltered_df: DataFrame = the full dataframe for the present training period
        :param training_feature_list: List[str] = the training feature list constructed by
                                      self.build_feature_list() according to user specified
                                      parameters in the configuration file.
        :param label_list: Optional[List[str]] = the labels for the dataset
        :param training_filter: bool = boolean which lets the function know if it is training data or
                                prediction data to be filtered.
        :returns:
        :filtered_df: DataFrame = dataframe cleaned of NaNs and only containing the user
        requested feature set.
        :labels: Union[DataFrame, List[Any]] = labels cleaned of NaNs or empty list
        """
        if label_list is None:
            label_list = []
        filtered_df: DataFrame = unfiltered_df.filter(training_feature_list, axis=1)
        filtered_df = filtered_df.replace([np.inf, -np.inf], np.nan)
        drop_index: pd.Series = pd.isnull(filtered_df).any(axis=1)
        drop_index = drop_index.replace(True, 1).replace(False, 0).infer_objects(copy=False)
        if training_filter:
            labels: DataFrame = unfiltered_df.filter(label_list, axis=1)
            drop_index_labels: pd.Series = pd.isnull(labels).any(axis=1)
            drop_index_labels = drop_index_labels.replace(True, 1).replace(False, 0).infer_objects(copy=False)
            dates: Series = unfiltered_df['date']
            filtered_df = filtered_df[(drop_index == 0) & (drop_index_labels == 0)]
            labels = labels[(drop_index == 0) & (drop_index_labels == 0)]
            self.train_dates = dates[(drop_index == 0) & (drop_index_labels == 0)]
            logger.info(
                f'{self.pair}: dropped {len(unfiltered_df) - len(filtered_df)} training points due to NaNs in populated dataset {len(unfiltered_df)}.'
            )
            if len(filtered_df) == 0 and not self.live:
                raise OperationalException(
                    f"""{self.pair}: all training data dropped due to NaNs. You likely did not download enough training data prior to your backtest timerange. Hint:
{DOCS_LINK}/freqai-running/#downloading-data-to-cover-the-full-backtest-period"""
                )
            if 1 - len(filtered_df) / len(unfiltered_df) > 0.1 and self.live:
                worst_indicator: str = str(unfiltered_df.count().idxmin())
                logger.warning(
                    f' {(1 - len(filtered_df) / len(unfiltered_df)) * 100:.0f} percent  of training data dropped due to NaNs, model may perform inconsistent with expectations. Verify {worst_indicator}'
                )
            self.data['filter_drop_index_training'] = drop_index
        else:
            drop_index = pd.isnull(filtered_df).any(axis=1)
            self.data['filter_drop_index_prediction'] = drop_index
            filtered_df.fillna(0, inplace=True)
            drop_index = ~drop_index
            self.do_predict = np.array(drop_index.replace(True, 1).replace(False, 0))
            if len(self.do_predict) - self.do_predict.sum() > 0:
                logger.info(
                    'dropped %s of %s prediction data points due to NaNs.',
                    len(self.do_predict) - self.do_predict.sum(), len(filtered_df)
                )
            labels = []
        return filtered_df, labels

    def func_nrz1zx3m(
        self, 
        train_df: DataFrame, 
        test_df: DataFrame, 
        train_labels: DataFrame, 
        test_labels: DataFrame,
        train_weights: npt.NDArray[Any], 
        test_weights: npt.NDArray[Any]
    ) -> Dict[str, Any]:
        """
        Build the data dictionary with training and testing data.

        :param train_df: DataFrame = training features
        :param test_df: DataFrame = testing features
        :param train_labels: DataFrame = training labels
        :param test_labels: DataFrame = testing labels
        :param train_weights: npt.NDArray[Any] = training weights
        :param test_weights: npt.NDArray[Any] = testing weights
        :return: Dict[str, Any] = data dictionary containing all splits
        """
        self.data_dictionary = {
            'train_features': train_df, 
            'test_features': test_df, 
            'train_labels': train_labels, 
            'test_labels': test_labels, 
            'train_weights': train_weights, 
            'test_weights': test_weights, 
            'train_dates': self.train_dates
        }
        return self.data_dictionary

    def func_5au0tulw(
        self, 
        tr: str, 
        train_split: int = 28, 
        bt_split: int = 7
    ) -> Tuple[List[TimeRange], List[TimeRange]]:
        """
        Function which takes a single time range (tr) and splits it
        into sub timeranges to train and backtest on based on user input
        tr: str, full timerange to train on
        train_split: the period length for the each training (days). Specified in user
        configuration file
        bt_split: the backtesting length (days). Specified in user configuration file
        """
        if not isinstance(train_split, int) or train_split < 1:
            raise OperationalException(
                f'train_period_days must be an integer greater than 0. Got {train_split}.'
            )
        train_period_days: int = train_split * SECONDS_IN_DAY
        bt_period: int = bt_split * SECONDS_IN_DAY
        full_timerange: TimeRange = TimeRange.parse_timerange(tr)
        config_timerange: TimeRange = TimeRange.parse_timerange(self.config['timerange'])
        if config_timerange.stopts == 0:
            config_timerange.stopts = int(datetime.now(tz=timezone.utc).timestamp())
        timerange_train: TimeRange = copy.deepcopy(full_timerange)
        timerange_backtest: TimeRange = copy.deepcopy(full_timerange)
        tr_training_list: List[str] = []
        tr_backtesting_list: List[str] = []
        tr_training_list_timerange: List[TimeRange] = []
        tr_backtesting_list_timerange: List[TimeRange] = []
        first: bool = True
        while True:
            if not first:
                timerange_train.startts = timerange_train.startts + int(bt_period)
            timerange_train.stopts = timerange_train.startts + train_period_days
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

    def func_arhql2n5(
        self, 
        timerange: TimeRange, 
        df: DataFrame
    ) -> DataFrame:
        """
        Given a full dataframe, extract the user desired window
        :param timerange: TimeRange = timerange object to extract from df
        :param df: DataFrame = Dataframe containing all candles to run the entire backtest. Here
                   it is sliced down to just the present training period.
        :return: DataFrame = sliced dataframe based on timerange
        """
        if not self.live:
            df = df.loc[
                (df['date'] >= timerange.startdt) & 
                (df['date'] < timerange.stopdt), :
            ]
        else:
            df = df.loc[df['date'] >= timerange.startdt, :]
        return df

    def func_9cevie8g(self, dataframe: DataFrame) -> None:
        """
        Find features in the strategy provided dataframe
        :param dataframe: DataFrame = strategy provided dataframe
        """
        column_names: pd.Index = dataframe.columns
        features: List[str] = [c for c in column_names if '%' in c]
        if not features:
            raise OperationalException('Could not find any features!')
        self.training_features_list = features

    def func_w6rclape(self, dataframe: DataFrame) -> None:
        """
        Find labels in the strategy provided dataframe
        :param dataframe: DataFrame = strategy provided dataframe
        """
        column_names: pd.Index = dataframe.columns
        labels: List[str] = [c for c in column_names if '&' in c]
        self.label_list = labels

    def func_ttullx9p(self, num_weights: int) -> npt.NDArray[Any]:
        """
        Set weights so that recent data is more heavily weighted during
        training than older data.
        :param num_weights: int = number of weights to generate
        :return: npt.NDArray[Any] = array of weights
        """
        wfactor: float = self.config['freqai']['feature_parameters']['weight_factor']
        weights: npt.NDArray[Any] = np.exp(-np.arange(num_weights) / (wfactor * num_weights))[:-1]
        return weights

    def func_nfjs0ia2(
        self, 
        predictions: DataFrame, 
        do_predict: npt.NDArray[Any], 
        dataframe_backtest: DataFrame
    ) -> DataFrame:
        """
        Get backtest prediction from current backtest period
        :param predictions: DataFrame = prediction results
        :param do_predict: npt.NDArray[Any] = array indicating which rows to predict
        :param dataframe_backtest: DataFrame = backtest dataframe
        :return: DataFrame = merged dataframe with predictions
        """
        append_df: DataFrame = DataFrame()
        for label in predictions.columns:
            append_df[label] = predictions[label]
            if append_df[label].dtype == object:
                continue
            if 'labels_mean' in self.data:
                append_df[f'{label}_mean'] = self.data['labels_mean'][label]
            if 'labels_std' in self.data:
                append_df[f'{label}_std'] = self.data['labels_std'][label]
        for extra_col in self.data['extra_returns_per_train']:
            append_df[f'{extra_col}'] = self.data['extra_returns_per_train'][extra_col]
        append_df['do_predict'] = do_predict
        if self.freqai_config['feature_parameters'].get('DI_threshold', 0) > 0:
            append_df['DI_values'] = self.DI_values
        user_cols: List[str] = [col for col in dataframe_backtest.columns if col.startswith('%%')]
        cols: List[str] = ['date']
        cols.extend(user_cols)
        dataframe_backtest.reset_index(drop=True, inplace=True)
        merged_df: DataFrame = pd.concat([dataframe_backtest[cols], append_df], axis=1)
        return merged_df

    def func_uz7nyiii(self, append_df: DataFrame) -> None:
        """
        Append backtest prediction from current backtest period to all previous periods
        :param append_df: DataFrame = dataframe to append
        """
        if self.full_df.empty:
            self.full_df = append_df
        else:
            self.full_df = pd.concat([self.full_df, append_df], axis=0, ignore_index=True)

    def func_ifge3wnf(self, dataframe: DataFrame) -> None:
        """
        Back fill values to before the backtesting range so that the dataframe matches size
        when it goes back to the strategy. These rows are not included in the backtest.
        :param dataframe: DataFrame = dataframe to process
        """
        to_keep: List[str] = [col for col in dataframe.columns if not col.startswith('&') and not col.startswith('%%')]
        self.return_dataframe: DataFrame = pd.merge(
            dataframe[to_keep], 
            self.full_df,
            how='left', 
            on='date'
        )
        self.return_dataframe[self.full_df.columns] = self.return_dataframe[self.full_df.columns].fillna(value=0)
        self.full_df = DataFrame()
        return

    def func_8pjk0l77(
        self, 
        backtest_tr: str, 
        backtest_period_days: int
    ) -> str:
        """
        Process backtest timerange and prepare model path
        :param backtest_tr: str = backtest timerange string
        :param backtest_period_days: int = backtest period in days
        :return: str = full timerange string
        """
        if not isinstance(backtest_period_days, int):
            raise OperationalException('backtest_period_days must be an integer')
        if backtest_period_days < 0:
            raise OperationalException('backtest_period_days must be positive')
        backtest_timerange: TimeRange = TimeRange.parse_timerange(backtest_tr)
        if backtest_timerange.stopts == 0:
            raise OperationalException(
                'FreqAI backtesting does not allow open ended timeranges. Please indicate the end date of your desired backtesting timerange.'
            )
        backtest_timerange.startts = backtest_timerange.startts - backtest_period_days * SECONDS_IN_DAY
        full_timerange: str = backtest_timerange.timerange_str
        config_path: Path = Path(self.config['config_files'][0])
        if not self.full_path.is_dir():
            self.full_path.mkdir(parents=True, exist_ok=True)
            shutil.copy(config_path.resolve(), Path(self.full_path / config_path.name))
        return full_timerange

    def func_hcjwb100(self, trained_timestamp: int) -> bool:
        """
        A model age checker to determine if the model is trustworthy based on user defined
        `expiration_hours` in the configuration file.
        :param trained_timestamp: int = The time of training for the most recent model.
        :return:
            bool = If the model is expired or not.
        """
        current_time: float = datetime.now(tz=timezone.utc).timestamp()
        elapsed_time: float = (current_time - trained_timestamp) / 3600
        max_time: float = self.freqai_config.get('expiration_hours', 0)
        if max_time > 0:
            return elapsed_time > max_time
        else:
            return False

    def func_20naxam2(
        self, 
        trained_timestamp: int
    ) -> Tuple[bool, TimeRange, TimeRange]:
        """
        Determine if retraining is needed and compute timeranges.
        :param trained_timestamp: int = The time of training for the most recent model.
        :return: Tuple[bool, TimeRange, TimeRange] = retrain flag, trained_timerange, data_load_timerange
        """
        current_time: float = datetime.now(tz=timezone.utc).timestamp()
        trained_timerange: TimeRange = TimeRange()
        data_load_timerange: TimeRange = TimeRange()
        timeframes: List[str] = self.freqai_config['feature_parameters'].get('include_timeframes', [])
        max_tf_seconds: int = 0
        for tf in timeframes:
            secs: int = timeframe_to_seconds(tf)
            if secs > max_tf_seconds:
                max_tf_seconds = secs
        max_period: int = self.config.get('startup_candle_count', 20) * 2
        additional_seconds: int = max_period * max_tf_seconds
        if trained_timestamp != 0:
            elapsed_time: float = (current_time - trained_timestamp) / SECONDS_IN_HOUR
            retrain: bool = elapsed_time > self.freqai_config.get('live_retrain_hours', 0)
            if retrain:
                trained_timerange.startts = int(current_time - self.freqai_config.get('train_period_days', 0) * SECONDS_IN_DAY)
                trained_timerange.stopts = int(current_time)
                data_load_timerange.startts = int(
                    current_time - self.freqai_config.get('train_period_days', 0) * SECONDS_IN_DAY - additional_seconds
                )
                data_load_timerange.stopts = int(current_time)
        else:
            trained_timerange.startts = int(current_time - self.freqai_config.get('train_period_days', 0) * SECONDS_IN_DAY)
            trained_timerange.stopts = int(current_time)
            data_load_timerange.startts = int(
                current_time - self.freqai_config.get('train_period_days', 0) * SECONDS_IN_DAY - additional_seconds
            )
            data_load_timerange.stopts = int(current_time)
            retrain = True
        return retrain, trained_timerange, data_load_timerange

    def func_bpps17gt(
        self, 
        pair: str, 
        timestamp_id: Any
    ) -> None:
        """
        Set data path and model filename based on pair and timestamp.
        :param pair: str = trading pair
        :param timestamp_id: Any = timestamp identifier
        """
        coin: str
        coin, _ = pair.split('/')
        self.data_path = Path(self.full_path / f"sub-train-{pair.split('/')[0]}_{timestamp_id}")
        self.model_filename = f"cb_{coin.lower()}_{timestamp_id}"

    def func_tdw3nni1(self) -> None:
        """
        Initialize all_pairs list with include_corr_pairlist and pair_whitelist.
        """
        self.all_pairs: List[str] = copy.deepcopy(
            self.freqai_config['feature_parameters'].get('include_corr_pairlist', [])
        )
        for pair in self.config.get('exchange', {}).get('pair_whitelist', []):
            if pair not in self.all_pairs:
                self.all_pairs.append(pair)

    def func_alf38v0c(
        self, 
        dataframe: DataFrame
    ) -> Dict[str, DataFrame]:
        """
        Find the columns of the dataframe corresponding to the corr_pairlist, save them
        in a dictionary to be reused and attached to other pairs.

        :param dataframe: DataFrame = fully populated dataframe (current pair + corr_pairs)
        :return: Dict[str, DataFrame] = dictionary of dataframes to be attached to other pairs
        """
        corr_dataframes: Dict[str, DataFrame] = {}
        pairs: List[str] = self.freqai_config['feature_parameters'].get('include_corr_pairlist', [])
        for pair in pairs:
            pair_clean: str = pair.replace(':', '')
            pair_cols: List[str] = [col for col in dataframe.columns if col.startswith('%') and f'{pair_clean}_' in col]
            if pair_cols:
                pair_cols.insert(0, 'date')
                corr_dataframes[pair] = dataframe.filter(pair_cols, axis=1)
        return corr_dataframes

    def func_psq937a0(
        self, 
        dataframe: DataFrame, 
        corr_dataframes: Dict[str, DataFrame], 
        current_pair: str
    ) -> DataFrame:
        """
        Attach the existing corr_pair dataframes to the current pair dataframe before training

        :param dataframe: DataFrame = current pair strategy dataframe, indicators populated already
        :param corr_dataframes: Dict[str, DataFrame] = dictionary of saved dataframes from earlier in the same candle
        :param current_pair: str = current pair to which we will attach corr pair dataframe
        :return: DataFrame = current pair dataframe with corr_pairs attached
        """
        pairs: List[str] = self.freqai_config['feature_parameters'].get('include_corr_pairlist', [])
        current_pair_clean: str = current_pair.replace(':', '')
        for pair in pairs:
            pair_clean: str = pair.replace(':', '')
            if current_pair_clean != pair_clean:
                dataframe = dataframe.merge(corr_dataframes[pair], how='left', on='date')
        return dataframe

    def func_ufxbr0cl(
        self, 
        pair: str, 
        tf: str, 
        strategy: IStrategy, 
        corr_dataframes: Optional[Dict[str, Dict[str, DataFrame]]] = None,
        base_dataframes: Optional[Dict[str, DataFrame]] = None,
        is_corr_pairs: bool = False
    ) -> DataFrame:
        """
        Get the data for the pair. If it's not in the dictionary, get it from the data provider
        :param pair: str = pair to get data for
        :param tf: str = timeframe to get data for
        :param strategy: IStrategy = user defined strategy object
        :param corr_dataframes: Optional[Dict[str, Dict[str, DataFrame]]] = dict containing the df pair dataframes
                                (for user defined timeframes)
        :param base_dataframes: Optional[Dict[str, DataFrame]] = dict containing the current pair dataframes
                                (for user defined timeframes)
        :param is_corr_pairs: bool = whether the pair is a corr pair or not
        :return: DataFrame = dataframe containing the pair data
        """
        if corr_dataframes is None:
            corr_dataframes = {}
        if base_dataframes is None:
            base_dataframes = {}
        if is_corr_pairs:
            dataframe: DataFrame = corr_dataframes[pair][tf]
            if not dataframe.empty:
                return dataframe
            else:
                dataframe = strategy.dp.get_pair_dataframe(pair=pair, timeframe=tf)
                return dataframe
        else:
            dataframe: DataFrame = base_dataframes[tf]
            if not dataframe.empty:
                return dataframe
            else:
                dataframe = strategy.dp.get_pair_dataframe(pair=pair, timeframe=tf)
                return dataframe

    def func_v9k00qzd(
        self, 
        df_main: DataFrame, 
        df_to_merge: DataFrame, 
        tf: str, 
        timeframe_inf: str, 
        suffix: str
    ) -> DataFrame:
        """
        Merge the features of the dataframe and remove HLCV and date added columns
        :param df_main: DataFrame = main dataframe
        :param df_to_merge: DataFrame = dataframe to merge
        :param tf: str = timeframe of the main dataframe
        :param timeframe_inf: str = timeframe of the dataframe to merge
        :param suffix: str = suffix to add to the columns of the dataframe to merge
        :return: DataFrame = merged dataframe
        """
        dataframe: DataFrame = merge_informative_pair(
            df_main, 
            df_to_merge, 
            tf,
            timeframe_inf=timeframe_inf, 
            append_timeframe=False, 
            suffix=suffix, 
            ffill=True
        )
        skip_columns: List[str] = [f"{s}_{suffix}" for s in ['date', 'open', 'high', 'low', 'close', 'volume']]
        dataframe = dataframe.drop(columns=skip_columns)
        return dataframe

    def func_30tumvq2(
        self, 
        dataframe: DataFrame, 
        pair: str, 
        strategy: IStrategy, 
        corr_dataframes: Dict[str, DataFrame],
        base_dataframes: Dict[str, DataFrame], 
        is_corr_pairs: bool = False
    ) -> DataFrame:
        """
        Use the user defined strategy functions for populating features
        :param dataframe: DataFrame = dataframe to populate
        :param pair: str = pair to populate
        :param strategy: IStrategy = user defined strategy object
        :param corr_dataframes: Dict[str, DataFrame] = dict containing the df pair dataframes
                                (for user defined timeframes)
        :param base_dataframes: Dict[str, DataFrame] = dict containing the current pair dataframes
                                (for user defined timeframes)
        :param is_corr_pairs: bool = whether the pair is a corr pair or not
        :return: DataFrame = populated dataframe
        """
        tfs: List[str] = self.freqai_config['feature_parameters'].get('include_timeframes', [])
        for tf in tfs:
            metadata: Dict[str, Any] = {'pair': pair, 't': tf}
            informative_df: DataFrame = self.get_pair_data_for_features(
                pair, tf, strategy, corr_dataframes, base_dataframes, is_corr_pairs
            )
            informative_copy: DataFrame = informative_df.copy()
            logger.debug(f'Populating features for {pair} {tf}')
            for t in self.freqai_config['feature_parameters']['indicator_periods_candles']:
                df_features: DataFrame = strategy.feature_engineering_expand_all(
                    informative_copy.copy(), t, metadata=metadata
                )
                suffix: str = f"{t}"
                informative_df = self.merge_features(
                    informative_df, 
                    df_features, 
                    tf, 
                    tf, 
                    suffix
                )
            generic_df: DataFrame = strategy.feature_engineering_expand_basic(
                informative_copy.copy(), metadata=metadata
            )
            suffix = 'gen'
            informative_df = self.merge_features(
                informative_df, 
                generic_df, 
                tf, 
                tf, 
                suffix
            )
            indicators: List[str] = [col for col in informative_df if col.startswith('%')]
            for n in range(self.freqai_config['feature_parameters'].get('include_shifted_candles', 0) + 1):
                if n == 0:
                    continue
                df_shift: DataFrame = informative_df[indicators].shift(n)
                df_shift = df_shift.add_suffix('_shift-' + str(n))
                informative_df = pd.concat((informative_df, df_shift), axis=1)
            dataframe = self.merge_features(
                dataframe.copy(),
                informative_df, 
                self.config['timeframe'], 
                tf, 
                f"{pair}_{tf}"
            )
        return dataframe

    def func_j5xkv503(
        self, 
        strategy: IStrategy, 
        corr_dataframes: Optional[Dict[str, Dict[str, DataFrame]]] = None, 
        base_dataframes: Optional[Dict[str, DataFrame]] = None,
        pair: str = '', 
        prediction_dataframe: DataFrame = pd.DataFrame(), 
        do_corr_pairs: bool = True
    ) -> DataFrame:
        """
        Use the user defined strategy for populating indicators during retrain
        :param strategy: IStrategy = user defined strategy object
        :param corr_dataframes: Optional[Dict[str, Dict[str, DataFrame]]] = dict containing the df pair dataframes
                                (for user defined timeframes)
        :param base_dataframes: Optional[Dict[str, DataFrame]] = dict containing the current pair dataframes
                                (for user defined timeframes)
        :param pair: str = pair to populate
        :param prediction_dataframe: DataFrame = dataframe containing the pair data
        used for prediction
        :param do_corr_pairs: bool = whether to populate corr pairs or not
        :return: DataFrame = dataframe containing populated indicators
        """
        if corr_dataframes is None:
            corr_dataframes = {}
        if base_dataframes is None:
            base_dataframes = {}
        new_version: bool = inspect.getsource(strategy.populate_any_indicators) == inspect.getsource(IStrategy.populate_any_indicators)
        if not new_version:
            raise OperationalException(
                f"""You are using the `populate_any_indicators()` function which was deprecated on March 1, 2023. Please refer to the strategy migration guide to use the new feature_engineering_* methods: 
{DOCS_LINK}/strategy_migration/#freqai-strategy 
And the feature_engineering_* documentation: 
{DOCS_LINK}/freqai-feature-engineering/"""
            )
        tfs: List[str] = self.freqai_config['feature_parameters'].get('include_timeframes', [])
        pairs: List[str] = self.freqai_config['feature_parameters'].get('include_corr_pairlist', [])
        for tf in tfs:
            if tf not in base_dataframes:
                base_dataframes[tf] = pd.DataFrame()
            for p in pairs:
                if p not in corr_dataframes:
                    corr_dataframes[p] = {}
                if tf not in corr_dataframes[p]:
                    corr_dataframes[p][tf] = pd.DataFrame()
        if not prediction_dataframe.empty:
            dataframe: DataFrame = prediction_dataframe.copy()
            base_dataframes[self.config['timeframe']] = dataframe.copy()
        else:
            dataframe = base_dataframes[self.config['timeframe']].copy()
        corr_pairs: List[str] = self.freqai_config['feature_parameters'].get('include_corr_pairlist', [])
        dataframe = self.populate_features(
            dataframe.copy(), 
            pair, 
            strategy, 
            corr_dataframes, 
            base_dataframes
        )
        metadata: Dict[str, Any] = {'pair': pair}
        dataframe = strategy.feature_engineering_standard(dataframe.copy(), metadata=metadata)
        for corr_pair in corr_pairs:
            if pair == corr_pair:
                continue
            if corr_pairs and do_corr_pairs:
                dataframe = self.populate_features(
                    dataframe.copy(),
                    corr_pair, 
                    strategy, 
                    corr_dataframes, 
                    base_dataframes, 
                    True
                )
        if self.live:
            dataframe = strategy.set_freqai_targets(dataframe.copy(), metadata=metadata)
            dataframe = self.remove_special_chars_from_feature_names(dataframe)
            self.get_unique_classes_from_labels(dataframe)
        if self.config.get('reduce_df_footprint', False):
            dataframe = reduce_dataframe_footprint(dataframe)
        return dataframe

    def func_lkfjli3o(self) -> None:
        """
        Fit the labels with a gaussian distribution
        """
        import scipy as spy
        self.data['labels_mean'], self.data['labels_std'] = {}, {}
        for label in self.data_dictionary['train_labels'].columns:
            if self.data_dictionary['train_labels'][label].dtype == object:
                continue
            f: Tuple[float, float] = spy.stats.norm.fit(self.data_dictionary['train_labels'][label])
            self.data['labels_mean'][label], self.data['labels_std'][label] = f[0], f[1]
        for label in self.unique_class_list:
            self.data['labels_mean'][label], self.data['labels_std'][label] = 0, 0
        return

    def func_rlve2pq3(self, dataframe: DataFrame) -> DataFrame:
        """
        Remove the features from the dataframe before returning it to strategy. This keeps it
        compact for Frequi purposes.
        :param dataframe: DataFrame = dataframe to process
        :return: DataFrame = trimmed dataframe
        """
        to_keep: List[str] = [col for col in dataframe.columns if not col.startswith('%') or col.startswith('%%')]
        return dataframe[to_keep]

    def func_pczf2gvr(self, dataframe: DataFrame) -> None:
        """
        Find labels in the dataframe and populate unique classes.
        :param dataframe: DataFrame = dataframe to process
        """
        self.find_labels(dataframe)
        for key in self.label_list:
            if dataframe[key].dtype == object:
                self.unique_classes[key] = dataframe[key].dropna().unique()
        if self.unique_classes:
            for label in self.unique_classes:
                self.unique_class_list += list(self.unique_classes[label])

    def func_bk0trlb0(self, append_df: DataFrame) -> None:
        """
        Save prediction dataframe from backtesting to feather file format
        :param append_df: DataFrame = dataframe for backtesting period
        """
        full_predictions_folder: Path = Path(self.full_path / self.backtest_predictions_folder)
        if not full_predictions_folder.is_dir():
            full_predictions_folder.mkdir(parents=True, exist_ok=True)
        append_df.to_feather(self.backtesting_results_path)

    def func_td1hdp1e(self) -> DataFrame:
        """
        Get prediction dataframe from feather file format
        :return: DataFrame = prediction dataframe
        """
        append_df: DataFrame = pd.read_feather(self.backtesting_results_path)
        return append_df

    def func_zt2twb8b(self, len_backtest_df: int) -> bool:
        """
        Check if a backtesting prediction already exists and if the predictions
        to append have the same size as the backtesting dataframe slice
        :param len_backtest_df: int = Length of backtesting dataframe slice
        :return: bool = whether the prediction file is valid.
        """
        path_to_predictionfile: Path = Path(
            self.full_path / self.backtest_predictions_folder / f"{self.model_filename}_prediction.feather"
        )
        self.backtesting_results_path = path_to_predictionfile
        file_exists: bool = path_to_predictionfile.is_file()
        if file_exists:
            append_df: DataFrame = self.get_backtesting_prediction()
            if len(append_df) == len_backtest_df and 'date' in append_df.columns:
                logger.info(f'Found backtesting prediction file at {path_to_predictionfile}')
                return True
            else:
                logger.info('A new backtesting prediction file is required. (Number of predictions is different from dataframe length or old prediction file version).')
                return False
        else:
            logger.info(f'Could not find backtesting prediction file at {path_to_predictionfile}')
            return False

    def func_lh4cgft7(self, config: Dict[str, Any]) -> Path:
        """
        Returns default FreqAI model path
        :param config: Configuration dictionary
        :return: Path = default model path
        """
        freqai_config: Dict[str, Any] = config['freqai']
        return Path(config['user_data_dir'] / 'models' / str(freqai_config.get('identifier')))

    def func_6sddamf9(self, dataframe: DataFrame) -> DataFrame:
        """
        Remove all special characters from feature strings (:)
        :param dataframe: DataFrame = the dataframe that just finished indicator population. (unfiltered)
        :return: DataFrame = dataframe with cleaned feature names
        """
        spec_chars: List[str] = [':']
        for c in spec_chars:
            dataframe.columns = dataframe.columns.str.replace(c, '')
        return dataframe

    def func_19c2q2h5(self, timerange: TimeRange) -> TimeRange:
        """
        Buffer the start and end of the timerange. This is used *after* the indicators
        are populated.

        The main example use is when predicting maxima and minima, the argrelextrema
        function  cannot know the maxima/minima at the edges of the timerange. To improve
        model accuracy, it is best to compute argrelextrema on the full timerange
        and then use this function to cut off the edges (buffer) by the kernel.

        In another case, if the targets are set to a shifted price movement, this
        buffer is unnecessary because the shifted candles at the end of the timerange
        will be NaN and FreqAI will automatically cut those off of the training
        dataset.
        :param timerange: TimeRange = original timerange
        :return: TimeRange = buffered timerange
        """
        buffer: int = self.freqai_config['feature_parameters']['buffer_train_data_candles']
        if buffer:
            timerange.stopts -= buffer * timeframe_to_seconds(self.config['timeframe'])
            timerange.startts += buffer * timeframe_to_seconds(self.config['timeframe'])
        return timerange

    def func_mcgdphsk(
        self, 
        data_dictionary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Deprecation warning, migration assistance
        :param data_dictionary: Dict[str, Any] = data dictionary
        :return: Dict[str, Any] = data dictionary
        """
        logger.warning(
            f'Your custom IFreqaiModel relies on the deprecated data pipeline. Please update your model to use the new data pipeline. This can be achieved by following the migration guide at {DOCS_LINK}/strategy_migration/#freqai-new-data-pipeline We added a basic pipeline for you, but this will be removed in a future version.'
        )
        return data_dictionary

    def func_d67v4it3(self, df: DataFrame) -> DataFrame:
        """
        Deprecation warning, migration assistance
        :param df: DataFrame = dataframe to process
        :return: DataFrame = inverse transformed prediction dataframe
        """
        logger.warning(
            f'Your custom IFreqaiModel relies on the deprecated data pipeline. Please update your model to use the new data pipeline. This can be achieved by following the migration guide at {DOCS_LINK}/strategy_migration/#freqai-new-data-pipeline We added a basic pipeline for you, but this will be removed in a future version.'
        )
        pred_df: DataFrame
        pred_df, _, _ = self.label_pipeline.inverse_transform(df)
        return pred_df
