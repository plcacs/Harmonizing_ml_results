from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, TypedDict
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

class pair_info(TypedDict):
    pass

class FreqaiDataDrawer:
    def __init__(self, full_path: Path, config: Config):
    def update_metric_tracker(self, metric: str, value: Any, pair: str):
    def collect_metrics(self, time_spent: float, pair: str):
    def load_global_metadata_from_disk(self) -> dict:
    def load_drawer_from_disk(self):
    def load_metric_tracker_from_disk(self):
    def load_historic_predictions_from_disk(self) -> bool:
    def save_historic_predictions_to_disk(self):
    def save_metric_tracker_to_disk(self):
    def save_drawer_to_disk(self):
    def save_global_metadata_to_disk(self, metadata: dict):
    def np_encoder(self, obj: Any):
    def get_pair_dict_info(self, pair: str) -> tuple:
    def set_pair_dict_info(self, metadata: dict):
    def set_initial_return_values(self, pair: str, pred_df: DataFrame, dataframe: DataFrame):
    def append_model_predictions(self, pair: str, predictions: DataFrame, do_preds: NDArray, dk: FreqaiDataKitchen, strat_df: DataFrame):
    def attach_return_values_to_return_dataframe(self, pair: str, dataframe: DataFrame) -> DataFrame:
    def return_null_values_to_strategy(self, dataframe: DataFrame, dk: FreqaiDataKitchen):
    def purge_old_models(self):
    def save_metadata(self, dk: FreqaiDataKitchen):
    def save_data(self, model: Any, coin: str, dk: FreqaiDataKitchen):
    def load_metadata(self, dk: FreqaiDataKitchen):
    def load_data(self, coin: str, dk: FreqaiDataKitchen) -> Any:
    def update_historic_data(self, strategy: IStrategy, dk: FreqaiDataKitchen):
    def load_all_pair_histories(self, timerange: TimeRange, dk: FreqaiDataKitchen):
    def get_base_and_corr_dataframes(self, timerange: TimeRange, pair: str, dk: FreqaiDataKitchen) -> tuple:
    def get_timerange_from_live_historic_predictions(self) -> TimeRange:
