from __future__ import annotations
import copy
import logging
import re
from datetime import datetime
from typing import Any, cast, ClassVar, TYPE_CHECKING, TypedDict, Dict, List, Optional, Tuple, Union
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
config = app.config
stats_logger: BaseStatsLogger = config['STATS_LOGGER']
logger: logging.Logger = logging.getLogger(__name__)
OFFSET_JOIN_COLUMN_SUFFIX: str = '__offset_join_column_'
AGGREGATED_JOIN_GRAINS: set = {TimeGrain.WEEK, TimeGrain.WEEK_STARTING_SUNDAY, TimeGrain.WEEK_STARTING_MONDAY, TimeGrain.WEEK_ENDING_SATURDAY, TimeGrain.WEEK_ENDING_SUNDAY, TimeGrain.MONTH, TimeGrain.QUARTER, TimeGrain.YEAR}
R_SUFFIX: str = '__right_suffix'

class CachedTimeOffset(TypedDict):
    df: pd.DataFrame
    queries: List[str]
    cache_keys: List[Optional[str]]

class QueryContextProcessor:
    """
    The query context contains the query object and additional fields necessary
    to retrieve the data payload for a given viz.
    """

    def __init__(self, query_context: QueryContext) -> None:
        self._query_context: QueryContext = query_context
        self._qc_datasource: Union[BaseDatasource, Query] = query_context.datasource
    cache_type: str = 'df'
    enforce_numerical_metrics: bool = True

    def get_df_payload(self, query_obj: QueryObject, force_cached: bool = False) -> Dict[str, Any]:
        """Handles caching around the df payload retrieval"""
        cache_key: Optional[str] = self.query_cache_key(query_obj)
        timeout: int = self.get_cache_timeout()
        force_query: bool = self._query_context.force or timeout == -1
        cache: QueryCacheManager = QueryCacheManager.get(key=cache_key, region=CacheRegion.DATA, force_query=force_query, force_cached=force_cached)
        if query_obj and cache_key and (not cache.is_loaded):
            try:
                if (invalid_columns := [col for col in get_column_names_from_columns(query_obj.columns) + get_column_names_from_metrics(query_obj.metrics or []) if col not in self._qc_datasource.column_names and col != DTTM_ALIAS]):
                    raise QueryObjectValidationError(_('Columns missing in dataset: %(invalid_columns)s', invalid_columns=invalid_columns))
                query_result: QueryResult = self.get_query_result(query_obj)
                annotation_data: Dict[str, Any] = self.get_annotation_data(query_obj)
                cache.set_query_result(key=cache_key, query_result=query_result, annotation_data=annotation_data, force_query=force_query, timeout=self.get_cache_timeout(), datasource_uid=self._qc_datasource.uid, region=CacheRegion.DATA)
            except QueryObjectValidationError as ex:
                cache.error_message = str(ex)
                cache.status = QueryStatus.FAILED
        label_map: Dict[str, List[str]] = {unescape_separator(col): [unescape_separator(col) for col in re.split('(?<!\\\\),\\s', col)] for col in cache.df.columns.values}
        cache.df.columns = [unescape_separator(col) for col in cache.df.columns.values]
        return {'cache_key': cache_key, 'cached_dttm': cache.cache_dttm, 'cache_timeout': self.get_cache_timeout(), 'df': cache.df, 'applied_template_filters': cache.applied_template_filters, 'applied_filter_columns': cache.applied_filter_columns, 'rejected_filter_columns': cache.rejected_filter_columns, 'annotation_data': cache.annotation_data, 'error': cache.error_message, 'is_cached': cache.is_cached, 'query': cache.query, 'status': cache.status, 'stacktrace': cache.stacktrace, 'rowcount': len(cache.df.index), 'sql_rowcount': cache.sql_rowcount, 'from_dttm': query_obj.from_dttm, 'to_dttm': query_obj.to_dttm, 'label_map': label_map}

    def query_cache_key(self, query_obj: QueryObject, **kwargs: Any) -> Optional[str]:
        """
        Returns a QueryObject cache key for objects in self.queries
        """
        datasource: Union[BaseDatasource, Query] = self._qc_datasource
        extra_cache_keys: Dict[str, Any] = datasource.get_extra_cache_keys(query_obj.to_dict())
        cache_key: Optional[str] = query_obj.cache_key(datasource=datasource.uid, extra_cache_keys=extra_cache_keys, rls=security_manager.get_rls_cache_key(datasource), changed_on=datasource.changed_on, **kwargs) if query_obj else None
        return cache_key

    def get_query_result(self, query_object: QueryObject) -> QueryResult:
        """Returns a pandas dataframe based on the query object"""
        query_context: QueryContext = self._query_context
        query: str = ''
        if isinstance(query_context.datasource, Query):
            result: QueryResult = query_context.datasource.exc_query(query_object.to_dict())
        else:
            result = query_context.datasource.query(query_object.to_dict())
            query = result.query + ';\n\n'
        df: pd.DataFrame = result.df
        if not df.empty:
            df = self.normalize_df(df, query_object)
            if query_object.time_offsets:
                time_offsets: CachedTimeOffset = self.processing_time_offsets(df, query_object)
                df = time_offsets['df']
                queries: List[str] = time_offsets['queries']
                query += ';\n\n'.join(queries)
                query += ';\n\n'
            try:
                df = query_object.exec_post_processing(df)
            except InvalidPostProcessingError as ex:
                raise QueryObjectValidationError(ex.message) from ex
        result.df = df
        result.query = query
        result.from_dttm = query_object.from_dttm
        result.to_dttm = query_object.to_dttm
        return result

    def normalize_df(self, df: pd.DataFrame, query_object: QueryObject) -> pd.DataFrame:

        def _get_timestamp_format(source: Union[BaseDatasource, Query], column: str) -> Optional[str]:
            column_obj: Any = source.get_column(column)
            if column_obj and hasattr(column_obj, 'python_date_format') and (formatter := column_obj.python_date_format):
                return str(formatter)
            return None
        datasource: Union[BaseDatasource, Query] = self._qc_datasource
        labels: Tuple[str, ...] = tuple((label for label in [*get_base_axis_labels(query_object.columns), query_object.granularity] if datasource and hasattr(datasource, 'get_column') and (col := datasource.get_column(label)) and (col.get('is_dttm') if isinstance(col, dict) else col.is_dttm)))
        dttm_cols: List[DateColumn] = [DateColumn(timestamp_format=_get_timestamp_format(datasource, label), offset=datasource.offset, time_shift=query_object.time_shift, col_label=label) for label in labels if label]
        if DTTM_ALIAS in df:
            dttm_cols.append(DateColumn.get_legacy_time_column(timestamp_format=_get_timestamp_format(datasource, query_object.granularity), offset=datasource.offset, time_shift=query_object.time_shift))
        normalize_dttm_col(df=df, dttm_cols=tuple(dttm_cols))
        if self.enforce_numerical_metrics:
            dataframe_utils.df_metrics_to_num(df, query_object)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        return df

    @staticmethod
    def get_time_grain(query_object: QueryObject) -> Optional[str]:
        if query_object.columns and len(query_object.columns) > 0 and isinstance(query_object.columns[0], dict):
            return query_object.columns[0].get('timeGrain')
        return query_object.extras.get('time_grain_sqla')

    def add_offset_join_column(self, df: pd.DataFrame, name: str, time_grain: Optional[str], time_offset: Optional[str] = None, join_column_producer: Optional[Any] = None) -> None:
        """
        Adds an offset join column to the provided DataFrame.

        The function modifies the DataFrame in-place.

        :param df: pandas DataFrame to which the offset join column will be added.
        :param name: The name of the new column to be added.
        :param time_grain: The time grain used to calculate the new column.
        :param time_offset: The time offset used to calculate the new column.
        :param join_column_producer: A function to generate the join column.
        """
        if join_column_producer:
            df[name] = df.apply(lambda row: join_column_producer(row, 0), axis=1)
        else:
            df[name] = df.apply(lambda row: self.generate_join_column(row, 0, time_grain, time_offset), axis=1)

    def is_valid_date(self, date_string: str) -> bool:
        try:
            datetime.strptime(date_string, '%Y-%m-%d')
            return True
        except ValueError:
            return False

    def get_offset_custom_or_inherit(self, offset: str, outer_from_dttm: datetime, outer_to_dttm: datetime) -> str:
        """
        Get the time offset for custom or inherit.

        :param offset: The offset string.
        :param outer_from_dttm: The outer from datetime.
        :param outer_to_dttm: The outer to datetime.
        :returns: The time offset.
        """
        if offset == 'inherit':
            return f'{(outer_to_dttm - outer_from_dttm).days} days ago'
        if self.is_valid_date(offset):
            offset_date = datetime.strptime(offset, '%Y-%m-%d')
            return f'{(outer_from_dttm - offset_date).days} days ago'
        return ''

    def processing_time_offsets(self, df: pd.DataFrame, query_object: QueryObject) -> CachedTimeOffset:
        query_context: QueryContext = self._query_context
        query_object_clone: QueryObject = copy.copy(query_object)
        queries: List[str] = []
        cache_keys: List[Optional[str]] = []
        offset_dfs: Dict[str, pd.DataFrame] = {}
        outer_from_dttm: Optional[datetime]
        outer_to_dttm: Optional[datetime]
        outer_from_dttm, outer_to_dttm = get_since_until_from_query_object(query_object)
        if not outer_from_dttm or not outer_to_dttm:
            raise QueryObjectValidationError(_('An enclosed time range (both start and end) must be specified when using a Time Comparison.'))
        time_grain: Optional[str] = self.get_time_grain(query_object)
        metric_names: List[str] = get_metric_names(query_object.metrics)
        join_keys: List[str] = [col for col in df.columns if col not in metric_names]
        for offset in query_object.time_offsets:
            try:
                original_offset: str = offset
                if self.is_valid_date(offset) or offset == 'inherit':
                    offset = self.get_offset_custom_or_inherit(offset, outer_from_dttm, outer_to_dttm)
                query_object_clone.from_dttm = get_past_or_future(offset, outer_from_dttm)
                query_object_clone.to_dttm = get_past_or_future(offset, outer_to_dttm)
                x_axis_label: str = get_x_axis_label(query_object.columns)
                query_object_clone.granularity = query_object_clone.granularity or x_axis_label
            except ValueError as ex:
                raise QueryObjectValidationError(str(ex)) from ex
            query_object_clone.inner_from_dttm = outer_from_dttm
            query_object_clone.inner_to_dttm = outer_to_dttm
            query_object_clone.time_offsets = []
            query_object_clone.post_processing = []
            index: str = (get_base_axis_labels(query_object.columns) or [DTTM_ALIAS])[0]
            if not dataframe_utils.is_datetime_series(df.get(index)):
                for flt in query_object_clone.filter:
                    if flt.get('op') == FilterOperator.TEMPORAL_RANGE.value and isinstance(flt.get('val'), str):
                        time_range: str = cast(str, flt.get('val'))
                        new_outer_from_dttm: datetime
                        new_outer_to_dttm: datetime
                        new_outer_from_dttm, new_outer_to_dttm = get_since_until_from_time_range(time_range=time_range, time_shift=offset)
                        flt['val'] = f'{new_outer_from_dttm} : {new_outer_to_dttm}'
            query_object_clone.filter = [flt for flt in query_object_clone.filter if flt.get('col') != x_axis_label]
            cached_time_offset_key: str = offset if offset == original_offset else f'{offset}_{original_offset}'
            cache_key: Optional[str] = self.query_cache_key(query_object_clone, time_offset=cached_time_offset_key, time_grain=time_grain)
            cache: QueryCacheManager = QueryCacheManager.get(cache_key, CacheRegion.DATA, query_context.force)
            if cache.is_loaded:
                offset_dfs[offset] = cache.df
                queries.append(cache.query)
                cache_keys.append(cache_key)
                continue
            query_object_clone_dct: Dict[str, Any] = query_object_clone.to_dict()
            metrics_mapping: Dict[str, str] = {metric: TIME_COMPARISON.join([metric, original_offset]) for metric in metric_names}
            if query_object.row_limit or query_object.row_offset:
                query_object_clone_dct['row_limit'] = config['ROW_LIMIT']
                query_object_clone_dct['row_offset'] = 0
            if isinstance(self._qc_datasource, Query):
                result: QueryResult = self._qc_datasource.exc_query(query_object_clone_dct)
            else:
                result = self._qc_datasource.query(query_object_clone_dct)
            queries.append(result.query)
            cache_keys.append(None)
            offset_metrics_df: pd.DataFrame = result.df
            if offset_metrics_df.empty:
                offset_metrics_df = pd.DataFrame({col: [np.NaN] for col in join_keys + list(metrics_mapping.values())})
            else:
                offset_metrics_df = self.normalize_df(offset_metrics_df, query_object_clone)
                offset_metrics_df = offset_metrics_df.rename(columns=metrics_mapping)
            value: Dict[str, Any] = {'df': offset_metrics_df, 'query': result.query}
            cache.set(key=cache_key, value=value, timeout=self.get_cache_timeout(), datasource_uid=query_context.datasource.uid, region=CacheRegion.DATA)
            offset_dfs[offset] = offset_metrics_df
        if offset_dfs:
            df = self.join_offset_dfs(df, offset_dfs, time_grain, join_keys)
        return CachedTimeOffset(df=df, queries=queries, cache_keys=cache_keys)

    def join_offset_dfs(self, df: pd.DataFrame, offset_dfs: Dict[str, pd.DataFrame], time_grain: Optional[str], join_keys: List[str]) -> pd.DataFrame:
        """
        Join offset DataFrames with the main DataFrame.

        :param df: The main DataFrame.
        :param offset_dfs: A list of offset DataFrames.
        :param time_grain: The time grain used to calculate the temporal join key.
        :param join_keys: The keys to join on.
        """
        join_column_producer: Optional[Any] = config['TIME_GRAIN_JOIN_COLUMN_PRODUCERS'].get(time_grain)
        if join_column_producer and (not time_grain):
            raise QueryObjectValidationError(_('Time Grain must be specified when using Time Shift.'))
        for offset, offset_df in offset_dfs.items():
            actual_join_keys: List[str] = join_keys
            if time_grain:
                column_name: str = OFFSET