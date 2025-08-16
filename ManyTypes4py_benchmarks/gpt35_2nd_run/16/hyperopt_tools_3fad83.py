import logging
from collections.abc import Iterator
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import rapidjson
from pandas import isna, json_normalize
from freqtrade.constants import FTHYPT_FILEVERSION, Config
from freqtrade.enums import HyperoptState
from freqtrade.exceptions import OperationalException
from freqtrade.misc import deep_merge_dicts, round_dict, safe_value_fallback2
from freqtrade.optimize.hyperopt_epoch_filters import hyperopt_filter_epochs

logger: logging.Logger = logging.getLogger(__name__)
NON_OPT_PARAM_APPENDIX: str = '  # value loaded from strategy'
HYPER_PARAMS_FILE_FORMAT: int = rapidjson.NM_NATIVE | rapidjson.NM_NAN

def hyperopt_serializer(x: Any) -> Any:
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return str(x)

class HyperoptStateContainer:
    state: HyperoptState = HyperoptState.OPTIMIZE

    @classmethod
    def set_state(cls, value: HyperoptState) -> None:
        cls.state = value

class HyperoptTools:

    @staticmethod
    def get_strategy_filename(config: Config, strategy_name: str) -> Path:
        ...

    @staticmethod
    def export_params(params: dict, strategy_name: str, filename: Path) -> None:
        ...

    @staticmethod
    def load_params(filename: Path) -> dict:
        ...

    @staticmethod
    def try_export_params(config: Config, strategy_name: str, params: dict) -> None:
        ...

    @staticmethod
    def has_space(config: Config, space: str) -> bool:
        ...

    @staticmethod
    def _read_results(results_file: Path, batch_size: int = 10) -> Iterator:
        ...

    @staticmethod
    def _test_hyperopt_results_exist(results_file: Path) -> bool:
        ...

    @staticmethod
    def load_filtered_results(results_file: Path, config: Config) -> tuple:
        ...

    @staticmethod
    def show_epoch_details(results: dict, total_epochs: int, print_json: bool, no_header: bool = False, header_str: str = None) -> None:
        ...

    @staticmethod
    def _params_update_for_json(result_dict: dict, params: dict, non_optimized: dict, space: str) -> None:
        ...

    @staticmethod
    def _params_pretty_print(params: dict, space: str, header: str, non_optimized: dict = None) -> None:
        ...

    @staticmethod
    def _space_params(params: dict, space: str, r: int = None) -> dict:
        ...

    @staticmethod
    def _pprint_dict(params: dict, non_optimized: dict, indent: int = 4) -> str:
        ...

    @staticmethod
    def is_best_loss(results: dict, current_best_loss: float) -> bool:
        ...

    @staticmethod
    def format_results_explanation_string(results_metrics: dict, stake_currency: str) -> str:
        ...

    @staticmethod
    def _format_explanation_string(results: dict, total_epochs: int) -> str:
        ...

    @staticmethod
    def export_csv_file(config: Config, results: dict, csv_file: Path) -> None:
        ...
