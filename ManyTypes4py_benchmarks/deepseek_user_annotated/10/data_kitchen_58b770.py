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


pd.set_option("future.no_silent_downcasting", True)

SECONDS_IN_DAY = 86400
SECONDS_IN_HOUR = 3600

logger = logging.getLogger(__name__)


class FreqaiDataKitchen:
    def __init__(
        self,
        config: Config,
        live: bool = False,
        pair: str = "",
    ) -> None:
        self.data: Dict[str, Any] = {}
        self.data_dictionary: Dict[str, DataFrame] = {}
        self.config: Config = config
        self.freqai_config: Dict[str, Any] = config["freqai"]
        self.full_df: DataFrame = DataFrame()
        self.append_df: DataFrame = DataFrame()
        self.data_path: Path = Path()
        self.label_list: List[str] = []
        self.training_features_list: List[str] = []
        self.model_filename: str = ""
        self.backtesting_results_path: Path = Path()
        self.backtest_predictions_folder: str = "backtesting_predictions"
        self.live: bool = live
        self.pair: str = pair
        self.keras: bool = self.freqai_config.get("keras", False)
        self.set_all_pairs()
        self.backtest_live_models: bool = config.get("freqai_backtest_live_models", False)
        self.feature_pipeline: Pipeline = Pipeline()
        self.label_pipeline: Pipeline = Pipeline()
        self.DI_values: npt.NDArray[np.float64] = np.array([])

        if not self.live:
            self.full_path: Path = self.get_full_models_path(self.config)

            if not self.backtest_live_models:
                self.full_timerange: str = self.create_fulltimerange(
                    self.config["timerange"], self.freqai_config.get("train_period_days", 0)
                )
                (self.training_timeranges, self.backtesting_timeranges) = self.split_timerange(
                    self.full_timerange,
                    config["freqai"]["train_period_days"],
                    config["freqai"]["backtest_period_days"],
                )

        self.data["extra_returns_per_train"] = self.freqai_config.get("extra_returns_per_train", {})
        if not self.freqai_config.get("data_kitchen_thread_count", 0):
            self.thread_count: int = max(int(psutil.cpu_count() * 2 - 2), 1)
        else:
            self.thread_count: int = self.freqai_config["data_kitchen_thread_count"]
        self.train_dates: DataFrame = pd.DataFrame()
        self.unique_classes: Dict[str, List[Any]] = {}
        self.unique_class_list: List[Any] = []
        self.backtest_live_models_data: Dict[str, Any] = {}

    def set_paths(
        self,
        pair: str,
        trained_timestamp: Optional[int] = None,
    ) -> None:
        self.full_path = self.get_full_models_path(self.config)
        self.data_path = Path(
            self.full_path / f"sub-train-{pair.split('/')[0]}_{trained_timestamp}"
        )
        return

    def make_train_test_datasets(
        self, filtered_dataframe: DataFrame, labels: DataFrame
    ) -> Dict[str, Union[DataFrame, npt.NDArray[np.float64]]]:
        feat_dict = self.freqai_config["feature_parameters"]

        if "shuffle" not in self.freqai_config["data_split_parameters"]:
            self.freqai_config["data_split_parameters"].update({"shuffle": False})

        weights: npt.NDArray[np.float64]
        if feat_dict.get("weight_factor", 0) > 0:
            weights = self.set_weights_higher_recent(len(filtered_dataframe))
        else:
            weights = np.ones(len(filtered_dataframe))

        if self.freqai_config.get("data_split_parameters", {}).get("test_size", 0.1) != 0:
            (
                train_features,
                test_features,
                train_labels,
                test_labels,
                train_weights,
                test_weights,
            ) = train_test_split(
                filtered_dataframe[: filtered_dataframe.shape[0]],
                labels,
                weights,
                **self.config["freqai"]["data_split_parameters"],
            )
        else:
            test_labels = np.zeros(2)
            test_features = pd.DataFrame()
            test_weights = np.zeros(2)
            train_features = filtered_dataframe
            train_labels = labels
            train_weights = weights

        if feat_dict["shuffle_after_split"]:
            rint1 = random.randint(0, 100)
            rint2 = random.randint(0, 100)
            train_features = train_features.sample(frac=1, random_state=rint1).reset_index(
                drop=True
            )
            train_labels = train_labels.sample(frac=1, random_state=rint1).reset_index(drop=True)
            train_weights = (
                pd.DataFrame(train_weights)
                .sample(frac=1, random_state=rint1)
                .reset_index(drop=True)
                .to_numpy()[:, 0]
            )
            test_features = test_features.sample(frac=1, random_state=rint2).reset_index(drop=True)
            test_labels = test_labels.sample(frac=1, random_state=rint2).reset_index(drop=True)
            test_weights = (
                pd.DataFrame(test_weights)
                .sample(frac=1, random_state=rint2)
                .reset_index(drop=True)
                .to_numpy()[:, 0]
            )

        if self.freqai_config["feature_parameters"].get("reverse_train_test_order", False):
            return self.build_data_dictionary(
                test_features,
                train_features,
                test_labels,
                train_labels,
                test_weights,
                train_weights,
            )
        else:
            return self.build_data_dictionary(
                train_features,
                test_features,
                train_labels,
                test_labels,
                train_weights,
                test_weights,
            )

    def filter_features(
        self,
        unfiltered_df: DataFrame,
        training_feature_list: List[str],
        label_list: List[str] = list(),
        training_filter: bool = True,
    ) -> Tuple[DataFrame, DataFrame]:
        filtered_df = unfiltered_df.filter(training_feature_list, axis=1)
        filtered_df = filtered_df.replace([np.inf, -np.inf], np.nan)

        drop_index = pd.isnull(filtered_df).any(axis=1)
        drop_index = drop_index.replace(True, 1).replace(False, 0).infer_objects(copy=False)
        if training_filter:
            labels = unfiltered_df.filter(label_list, axis=1)
            drop_index_labels = pd.isnull(labels).any(axis=1)
            drop_index_labels = (
                drop_index_labels.replace(True, 1).replace(False, 0).infer_objects(copy=False)
            )
            dates = unfiltered_df["date"]
            filtered_df = filtered_df[
                (drop_index == 0) & (drop_index_labels == 0)
            ]
            labels = labels[
                (drop_index == 0) & (drop_index_labels == 0)
            ]
            self.train_dates = dates[(drop_index == 0) & (drop_index_labels == 0)]
            logger.info(
                f"{self.pair}: dropped {len(unfiltered_df) - len(filtered_df)} training points"
                f" due to NaNs in populated dataset {len(unfiltered_df)}."
            )
            if len(filtered_df) == 0 and not self.live:
                raise OperationalException(
                    f"{self.pair}: all training data dropped due to NaNs. "
                    "You likely did not download enough training data prior "
                    "to your backtest timerange. Hint:\n"
                    f"{DOCS_LINK}/freqai-running/"
                    "#downloading-data-to-cover-the-full-backtest-period"
                )
            if (1 - len(filtered_df) / len(unfiltered_df)) > 0.1 and self.live:
                worst_indicator = str(unfiltered_df.count().idxmin())
                logger.warning(
                    f" {(1 - len(filtered_df) / len(unfiltered_df)) * 100:.0f} percent "
                    " of training data dropped due to NaNs, model may perform inconsistent "
                    f"with expectations. Verify {worst_indicator}"
                )
            self.data["filter_drop_index_training"] = drop_index

        else:
            drop_index = pd.isnull(filtered_df).any(axis=1)
            self.data["filter_drop_index_prediction"] = drop_index
            filtered_df.fillna(0, inplace=True)
            drop_index = ~drop_index
            self.do_predict = np.array(drop_index.replace(True, 1).replace(False, 0))
            if (len(self.do_predict) - self.do_predict.sum()) > 0:
                logger.info(
                    "dropped %s of %s prediction data points due to NaNs.",
                    len(self.do_predict) - self.do_predict.sum(),
                    len(filtered_df),
                )
            labels = []

        return filtered_df, labels

    def build_data_dictionary(
        self,
        train_df: DataFrame,
        test_df: DataFrame,
        train_labels: DataFrame,
        test_labels: DataFrame,
        train_weights: npt.NDArray[np.float64],
        test_weights: npt.NDArray[np.float64],
    ) -> Dict[str, Union[DataFrame, npt.NDArray[np.float64]]]:
        self.data_dictionary = {
            "train_features": train_df,
            "test_features": test_df,
            "train_labels": train_labels,
            "test_labels": test_labels,
            "train_weights": train_weights,
            "test_weights": test_weights,
            "train_dates": self.train_dates,
        }
        return self.data_dictionary

    def split_timerange(
        self, tr: str, train_split: int = 28, bt_split: float = 7
    ) -> Tuple[List[TimeRange], List[TimeRange]]:
        if not isinstance(train_split, int) or train_split < 1:
            raise OperationalException(
                f"train_period_days must be an integer greater than 0. Got {train_split}."
            )
        train_period_days = train_split * SECONDS_IN_DAY
        bt_period = bt_split * SECONDS_IN_DAY

        full_timerange = TimeRange.parse_timerange(tr)
        config_timerange = TimeRange.parse_timerange(self.config["timerange"])
        if config_timerange.stopts == 0:
            config_timerange.stopts = int(datetime.now(tz=timezone.utc).timestamp())
        timerange_train = copy.deepcopy(full_timerange)
        timerange_backtest = copy.deepcopy(full_timerange)

        tr_training_list: List[TimeRange] = []
        tr_backtesting_list: List[TimeRange] = []
        first = True

        while True:
            if not first:
                timerange_train.startts = timerange_train.startts + int(bt_period)
            timerange_train.stopts = timerange_train.startts + train_period_days

            first = False
            tr_training_list.append(copy.deepcopy(timerange_train))

            timerange_backtest.startts = timerange_train.stopts
            timerange_backtest.stopts = timerange_backtest.startts + int(bt_period)

            if timerange_backtest.stopts > config_timerange.stopts:
                timerange_backtest.stopts = config_timerange.stopts

            tr_backtesting_list.append(copy.deepcopy(timerange_backtest))

            if timerange_backtest.stopts == config_timerange.stopts:
                break

        return tr_training_list, tr_backtesting_list

    def slice_dataframe(self, timerange: TimeRange, df: DataFrame) -> DataFrame:
        if not self.live:
            df = df.loc[(df["date"] >= timerange.startdt) & (df["date"] < timerange.stopdt), :]
        else:
            df = df.loc[df["date"] >= timerange.startdt, :]
        return df

    def find_features(self, dataframe: DataFrame) -> None:
        column_names = dataframe.columns
        features = [c for c in column_names if "%" in c]

        if not features:
            raise OperationalException("Could not find any features!")

        self.training_features_list = features

    def find_labels(self, dataframe: DataFrame) -> None:
        column_names = dataframe.columns
        labels = [c for c in column_names if "&" in c]
        self.label_list = labels

    def set_weights_higher_recent(self, num_weights: int) -> npt.NDArray[np.float64]:
        wfactor = self.config["freqai"]["feature_parameters"]["weight_factor"]
        weights = np.exp(-np.arange(num_weights) / (wfactor * num_weights))[::-1]
        return weights

    def get_predictions_to_append(
        self, predictions: DataFrame, do_predict: npt.NDArray[np.bool_], dataframe_backtest: DataFrame
    ) -> DataFrame:
        append_df = DataFrame()
        for label in predictions.columns:
            append_df[label] = predictions[label]
            if append_df[label].dtype == object:
                continue
            if "labels_mean" in self.data:
                append_df[f"{label}_mean"] = self.data["labels_mean"][label]
            if "labels_std" in self.data:
                append_df[f"{label}_std"] = self.data["labels_std"][label]

        for extra_col in self.data["extra_returns_per_train"]:
            append_df[f"{extra_col}"] = self.data["extra_returns_per_train"][extra_col]

        append_df["do_predict"] = do_predict
        if self.freqai_config["feature_parameters"].get("DI_threshold", 0) > 0:
            append_df["DI_values"] = self.DI_values

        user_cols = [col for col in dataframe_backtest.columns if col.startswith("%%")]
        cols = ["date"]
        cols.extend(user_cols)

        dataframe_backtest.reset_index(drop=True, inplace=True)
        merged_df = pd.concat([dataframe_backtest[cols], append_df], axis=1)
        return merged_df

    def append_predictions(self, append_df: DataFrame) -> None:
        if self.full_df.empty:
            self.full_df = append_df
        else:
            self.full_df = pd.concat([self.full_df, append_df], axis=0, ignore_index=True)

    def fill_predictions(self, dataframe: DataFrame) -> None:
        to_keep = [
            col for col in dataframe.columns if not col.startswith("&") and not col.startswith("%%")
        ]
        self.return_dataframe = pd.merge(dataframe[to_keep], self.full_df, how="left", on="date")
        self.return_dataframe[self.full_df.columns] = self.return_dataframe[
            self.full_df.columns
        ].fillna(value=0)
        self.full_df = DataFrame()
        return

    def create_fulltimerange(self, backtest_tr: str, backtest_period_days: int) -> str:
        if not isinstance(backtest_period_days, int):
            raise OperationalException("backtest_period_days must be an integer")

        if backtest_period_days < 0:
            raise OperationalException("backtest_period_days must be positive")

        backtest_timerange = TimeRange.parse_timerange(backtest_tr)

        if backtest_timerange.stopts == 0:
            raise OperationalException(
                "FreqAI backtesting does not allow open ended timeranges. "
                "Please indicate the end date of your desired backtesting. "
                "timerange."
            )

        backtest_timerange.startts = (
            backtest_timerange.startts - backtest_period_days * SECONDS_IN_DAY
        )
        full_timerange = backtest_timerange.timerange_str
        config_path = Path(self.config["config_files"][0])

        if not self.full_path.is_dir():
            self.full_path.mkdir(parents=True, exist_ok=True)
            shutil.copy(
                config_path.resolve(),
                Path(self.full_path / config_path.parts[-1]),
            )

        return full_timerange

    def check_if_model_expired(self, trained_timestamp: int) -> bool:
        time = datetime.now(tz=timezone.utc).timestamp()
        elapsed_time = (time - trained_timestamp) / 3600
        max_time = self.freqai_config.get("expiration_hours", 0)
        if max_time > 0:
            return elapsed_time > max_time
        else:
            return False

    def check_if_new_training_required(
        self, trained_timestamp: int
    ) -> Tuple[bool, TimeRange, TimeRange]:
        time = datetime.now(tz=timezone.utc).timestamp()
        trained_timerange = TimeRange()
        data_load_timerange = TimeRange()

        timeframes = self.freqai_config["feature_parameters"].get("include_timeframes")

        max_tf_seconds = 0
        for tf in timeframes:
            secs = timeframe_to_seconds(tf)
            if secs > max_tf_seconds:
                max_tf_seconds = secs

        max_period = self.config.get("startup_candle_count", 20) * 2
        additional_seconds =