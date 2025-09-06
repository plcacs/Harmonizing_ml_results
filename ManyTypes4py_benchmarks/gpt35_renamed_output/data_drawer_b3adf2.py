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

logger: logging.Logger = logging.getLogger(__name__)

FEATURE_PIPELINE: str = 'feature_pipeline'
LABEL_PIPELINE: str = 'label_pipeline'
TRAINDF: str = 'trained_df'
METADATA: str = 'metadata'
METADATA_NUMBER_MODE: int = rapidjson.NM_NATIVE | rapidjson.NM_NAN

class pair_info(TypedDict):
    pass

class FreqaiDataDrawer:
    def __init__(self, full_path: Path, config: Config) -> None:
        self.config: Config = config
        self.freqai_info: dict = config.get('freqai', {})
        self.pair_dict: dict = {}
        self.model_dictionary: dict = {}
        self.meta_data_dictionary: dict = {}
        self.model_return_values: dict = {}
        self.historic_data: dict = {}
        self.historic_predictions: dict = {}
        self.full_path: Path = full_path
        self.historic_predictions_path: Path = Path(self.full_path / 'historic_predictions.pkl')
        self.historic_predictions_bkp_path: Path = Path(self.full_path / 'historic_predictions.backup.pkl')
        self.pair_dictionary_path: Path = Path(self.full_path / 'pair_dictionary.json')
        self.global_metadata_path: Path = Path(self.full_path / 'global_metadata.json')
        self.metric_tracker_path: Path = Path(self.full_path / 'metric_tracker.json')
        self.load_drawer_from_disk()
        self.load_historic_predictions_from_disk()
        self.metric_tracker: dict = {}
        self.load_metric_tracker_from_disk()
        self.training_queue: dict = {}
        self.history_lock: threading.Lock = threading.Lock()
        self.save_lock: threading.Lock = threading.Lock()
        self.pair_dict_lock: threading.Lock = threading.Lock()
        self.metric_tracker_lock: threading.Lock = threading.Lock()
        self.old_DBSCAN_eps: dict = {}
        self.empty_pair_dict: dict = {'model_filename': '', 'trained_timestamp': 0, 'data_path': '', 'extras': {}}
        self.model_type: str = self.freqai_info.get('model_save_type', 'joblib')

    def update_metric_tracker(self, metric: str, value: Any, pair: str) -> None:
        ...

    def collect_metrics(self, time_spent: float, pair: str) -> None:
        ...

    def load_global_metadata_from_disk(self) -> dict:
        ...

    def load_drawer_from_disk(self) -> None:
        ...

    def load_metric_tracker_from_disk(self) -> None:
        ...

    def load_historic_predictions_from_disk(self) -> bool:
        ...

    def save_historic_predictions_to_disk(self) -> None:
        ...

    def save_metric_tracker_to_disk(self) -> None:
        ...

    def save_drawer_to_disk(self) -> None:
        ...

    def save_global_metadata_to_disk(self, metadata: dict) -> None:
        ...

    def np_encoder(self, obj: Any) -> Any:
        ...

    def get_pair_dict_info(self, pair: str) -> tuple:
        ...

    def set_pair_dict_info(self, metadata: dict) -> None:
        ...

    def set_initial_return_values(self, pair: str, pred_df: DataFrame, dataframe: DataFrame) -> None:
        ...

    def append_model_predictions(self, pair: str, predictions: DataFrame, do_preds: NDArray, dk: FreqaiDataKitchen, strat_df: DataFrame) -> None:
        ...

    def attach_return_values_to_return_dataframe(self, pair: str, dataframe: DataFrame) -> DataFrame:
        ...

    def return_null_values_to_strategy(self, dataframe: DataFrame, dk: FreqaiDataKitchen) -> None:
        ...

    def purge_old_models(self) -> None:
        ...

    def save_metadata(self, dk: FreqaiDataKitchen) -> None:
        ...

    def save_data(self, model: Any, coin: str, dk: FreqaiDataKitchen) -> None:
        ...

    def load_metadata(self, dk: FreqaiDataKitchen) -> None:
        ...

    def load_data(self, coin: str, dk: FreqaiDataKitchen) -> Any:
        ...

    def update_historic_data(self, strategy: IStrategy, dk: FreqaiDataKitchen) -> None:
        ...

    def load_all_pair_histories(self, timerange: TimeRange, dk: FreqaiDataKitchen) -> None:
        ...

    def get_base_and_corr_dataframes(self, timerange: TimeRange, pair: str, dk: FreqaiDataKitchen) -> tuple:
        ...

    def get_timerange_from_live_historic_predictions(self) -> TimeRange:
        ...
