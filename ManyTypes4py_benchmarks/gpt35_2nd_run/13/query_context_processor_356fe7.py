from __future__ import annotations
import copy
import logging
import re
from datetime import datetime
from typing import Any, cast, ClassVar, TYPE_CHECKING, TypedDict
import numpy as np
import pandas as pd
from flask_babel import gettext as _
from pandas import DateOffset
from superset import app
from superset.common.chart_data import ChartDataResultFormat
from superset.common.db_query_status import QueryStatus
from superset.common.query_actions import get_query_results
from superset.common.utils import dataframe_utils
from superset.common.utils.query_cache_manager import QueryCacheManager
from superset.common.utils.time_range_utils import get_since_until_from_query_object, get_since_until_from_time_range
from superset.connectors.sqla.models import BaseDatasource
from superset.constants import CacheRegion, TimeGrain
from superset.daos.annotation_layer import AnnotationLayerDAO
from superset.daos.chart import ChartDAO
from superset.exceptions import InvalidPostProcessingError, QueryObjectValidationError, SupersetException
from superset.extensions import cache_manager, security_manager
from superset.models.helpers import QueryResult
from superset.models.sql_lab import Query
from superset.utils import csv, excel
from superset.utils.cache import generate_cache_key, set_and_log_cache
from superset.utils.core import DatasourceType, DateColumn, DTTM_ALIAS, error_msg_from_exception, FilterOperator, GenericDataType, get_base_axis_labels, get_column_names_from_columns, get_column_names_from_metrics, get_metric_names, get_x_axis_label, normalize_dttm_col, TIME_COMPARISON
from superset.utils.date_parser import get_past_or_future, normalize_time_delta
from superset.utils.pandas_postprocessing.utils import unescape_separator
from superset.views.utils import get_viz
from superset.viz import viz_types

if TYPE_CHECKING:
    from superset.common.query_context import QueryContext
    from superset.common.query_object import QueryObject
    from superset.stats_logger import BaseStatsLogger

config: dict = app.config
stats_logger: BaseStatsLogger = config['STATS_LOGGER']
logger: logging.Logger = logging.getLogger(__name__)
OFFSET_JOIN_COLUMN_SUFFIX: str = '__offset_join_column_'
AGGREGATED_JOIN_GRAINS: set[TimeGrain] = {TimeGrain.WEEK, TimeGrain.WEEK_STARTING_SUNDAY, TimeGrain.WEEK_STARTING_MONDAY, TimeGrain.WEEK_ENDING_SATURDAY, TimeGrain.WEEK_ENDING_SUNDAY, TimeGrain.MONTH, TimeGrain.QUARTER, TimeGrain.YEAR}
R_SUFFIX: str = '__right_suffix'

class CachedTimeOffset(TypedDict):
    df: pd.DataFrame
    queries: list[str]
    cache_keys: list[str]

class QueryContextProcessor:
    _query_context: QueryContext

    def __init__(self, query_context: QueryContext) -> None:
        self._query_context = query_context
        self._qc_datasource = query_context.datasource
    cache_type: str = 'df'
    enforce_numerical_metrics: bool = True

    def get_df_payload(self, query_obj: QueryObject, force_cached: bool = False) -> dict:
        ...

    def query_cache_key(self, query_obj: QueryObject, **kwargs: Any) -> str:
        ...

    def get_query_result(self, query_object: QueryObject) -> QueryResult:
        ...

    def normalize_df(self, df: pd.DataFrame, query_object: QueryObject) -> pd.DataFrame:
        ...

    @staticmethod
    def get_time_grain(query_object: QueryObject) -> Any:
        ...

    def add_offset_join_column(self, df: pd.DataFrame, name: str, time_grain: TimeGrain, time_offset: str = None, join_column_producer: Any = None) -> None:
        ...

    def is_valid_date(self, date_string: str) -> bool:
        ...

    def get_offset_custom_or_inherit(self, offset: str, outer_from_dttm: datetime, outer_to_dttm: datetime) -> str:
        ...

    def processing_time_offsets(self, df: pd.DataFrame, query_object: QueryObject) -> CachedTimeOffset:
        ...

    def join_offset_dfs(self, df: pd.DataFrame, offset_dfs: dict[str, pd.DataFrame], time_grain: TimeGrain, join_keys: list[str]) -> pd.DataFrame:
        ...

    @staticmethod
    def generate_join_column(row: pd.Series, column_index: int, time_grain: TimeGrain, time_offset: str = None) -> str:
        ...

    def get_data(self, df: pd.DataFrame, coltypes: dict[str, Any]) -> Any:
        ...

    def get_payload(self, cache_query_context: bool = False, force_cached: bool = False) -> dict:
        ...

    def get_cache_timeout(self) -> int:
        ...

    def cache_key(self, **extra: Any) -> str:
        ...

    def get_annotation_data(self, query_obj: QueryObject) -> dict:
        ...

    @staticmethod
    def get_native_annotation_data(query_obj: QueryObject) -> dict:
        ...

    @staticmethod
    def get_viz_annotation_data(annotation_layer: dict, force: bool) -> dict:
        ...

    def raise_for_access(self) -> None:
        ...
