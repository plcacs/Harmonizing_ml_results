"""This module contains the 'Viz' objects

These objects represent the backend of all the visualizations that
Superset can render.
"""
from __future__ import annotations
import copy
import dataclasses
import logging
import math
from collections import defaultdict, OrderedDict
from datetime import datetime, timedelta
from itertools import product
from typing import Any, cast, Optional, TYPE_CHECKING, List, Dict, Tuple, Union
import geohash
import numpy as np
import pandas as pd
import polyline
from dateutil import relativedelta as rdelta
from deprecation import deprecated
from flask import request
from flask_babel import lazy_gettext as _
from geopy.point import Point
from pandas.tseries.frequencies import to_offset
from superset import app
from superset.common.db_query_status import QueryStatus
from superset.errors import ErrorLevel, SupersetError, SupersetErrorType
from superset.exceptions import (
    CacheLoadError,
    NullValueException,
    QueryObjectValidationError,
    SpatialException,
    SupersetSecurityException,
)
from superset.extensions import cache_manager, security_manager
from superset.models.helpers import QueryResult
from superset.sql_parse import sanitize_clause
from superset.superset_typing import (
    Column,
    Metric,
    QueryObjectDict,
    VizData,
    VizPayload,
)
from superset.utils import core as utils, csv, json
from superset.utils.cache import set_and_log_cache
from superset.utils.core import (
    apply_max_row_limit,
    DateColumn,
    DTTM_ALIAS,
    ExtraFiltersReasonType,
    get_column_name,
    get_column_names,
    get_column_names_from_columns,
    JS_MAX_INTEGER,
    merge_extra_filters,
    simple_filter_to_adhoc,
)
from superset.utils.date_parser import get_since_until, parse_past_timedelta
from superset.utils.hashing import md5_sha_from_str
if TYPE_CHECKING:
    from superset.connectors.sqla.models import BaseDatasource

config: Any = app.config
stats_logger: Any = config['STATS_LOGGER']
relative_start: Any = config['DEFAULT_RELATIVE_START_TIME']
relative_end: Any = config['DEFAULT_RELATIVE_END_TIME']
logger: logging.Logger = logging.getLogger(__name__)
METRIC_KEYS: List[str] = [
    'metric',
    'metrics',
    'percent_metrics',
    'metric_2',
    'secondary_metric',
    'x',
    'y',
    'size',
]

class BaseViz:
    """All visualizations derive this base class"""
    viz_type: Optional[str] = None
    verbose_name: str = 'Base Viz'
    credits: str = ''
    is_timeseries: bool = False
    cache_type: str = 'df'
    enforce_numerical_metrics: bool = True

    @deprecated(deprecated_in='3.0')
    def __init__(
        self,
        datasource: Optional[BaseDatasource],
        form_data: Dict[str, Any],
        force: bool = False,
        force_cached: bool = False,
    ) -> None:
        if not datasource:
            raise QueryObjectValidationError(_('Viz is missing a datasource'))
        self.datasource: BaseDatasource = datasource
        self.request: Any = request
        self.viz_type: Optional[str] = form_data.get('viz_type')
        self.form_data: Dict[str, Any] = form_data
        self.query: str = ''
        self.token: str = utils.get_form_data_token(form_data)
        self.groupby: List[Any] = self.form_data.get('groupby') or []
        self.time_shift: timedelta = timedelta()
        self.status: Optional[QueryStatus] = None
        self.error_msg: str = ''
        self.results: Optional[QueryResult] = None
        self.applied_filter_columns: List[Any] = []
        self.rejected_filter_columns: List[Any] = []
        self.errors: List[Dict[str, Any]] = []
        self.force: bool = force
        self._force_cached: bool = force_cached
        self.from_dttm: Optional[datetime] = None
        self.to_dttm: Optional[datetime] = None
        self._extra_chart_data: List[Tuple[str, pd.DataFrame]] = []
        self.process_metrics()
        self.applied_filters: List[Any] = []
        self.rejected_filters: List[Any] = []

    @property
    @deprecated(deprecated_in='3.0')
    def force_cached(self) -> bool:
        return self._force_cached

    @deprecated(deprecated_in='3.0')
    def process_metrics(self) -> None:
        self.metric_dict: OrderedDict[str, Any] = OrderedDict()
        for mkey in METRIC_KEYS:
            val = self.form_data.get(mkey)
            if val:
                if not isinstance(val, list):
                    val = [val]
                for o in val:
                    label = utils.get_metric_name(o)
                    self.metric_dict[label] = o
        self.all_metrics: List[Any] = list(self.metric_dict.values())
        self.metric_labels: List[str] = list(self.metric_dict.keys())

    @staticmethod
    @deprecated(deprecated_in='3.0')
    def handle_js_int_overflow(data: Dict[str, Any]) -> Dict[str, Any]:
        for record in data.get('records', {}):
            for k, v in list(record.items()):
                if isinstance(v, int):
                    if abs(v) > JS_MAX_INTEGER:
                        record[k] = str(v)
        return data

    @deprecated(deprecated_in='3.0')
    def run_extra_queries(self) -> None:
        """Lifecycle method to use when more than one query is needed

        In rare-ish cases, a visualization may need to execute multiple
        queries. That is the case for FilterBox or for time comparison
        in Line chart for instance.

        In those cases, we need to make sure these queries run before the
        main `get_payload` method gets called, so that the overall caching
        metadata can be right. The way it works here is that if any of
        the previous `get_df_payload` calls hit the cache, the main
        payload's metadata will reflect that.

        The multi-query support may need more work to become a first class
        use case in the framework, and for the UI to reflect the subtleties
        (show that only some of the queries were served from cache for
        instance). In the meantime, since multi-query is rare, we treat
        it with a bit of a hack. Note that the hack became necessary
        when moving from caching the visualization's data itself, to caching
        the underlying query(ies).
        """
        pass

    @deprecated(deprecated_in='3.0')
    def apply_rolling(self, df: pd.DataFrame) -> pd.DataFrame:
        rolling_type: Optional[str] = self.form_data.get('rolling_type')
        rolling_periods: int = int(self.form_data.get('rolling_periods') or 0)
        min_periods: int = int(self.form_data.get('min_periods') or 0)
        if rolling_type in ('mean', 'std', 'sum') and rolling_periods:
            kwargs: Dict[str, Any] = {'window': rolling_periods, 'min_periods': min_periods}
            if rolling_type == 'mean':
                df = df.rolling(**kwargs).mean()
            elif rolling_type == 'std':
                df = df.rolling(**kwargs).std()
            elif rolling_type == 'sum':
                df = df.rolling(**kwargs).sum()
        elif rolling_type == 'cumsum':
            df = df.cumsum()
        if min_periods:
            df = df[min_periods:]
        if df.empty:
            raise QueryObjectValidationError(
                _('Applied rolling window did not return any data. Please make sure the source query satisfies the minimum periods defined in the rolling window.')
            )
        return df

    @deprecated(deprecated_in='3.0')
    def get_samples(self) -> Dict[str, Any]:
        query_obj: Dict[str, Any] = self.query_obj()
        query_obj.update({
            'is_timeseries': False,
            'groupby': [],
            'metrics': [],
            'orderby': [],
            'row_limit': config['SAMPLES_ROW_LIMIT'],
            'columns': [o.column_name for o in self.datasource.columns],
            'from_dttm': None,
            'to_dttm': None
        })
        payload: Dict[str, Any] = self.get_df_payload(query_obj)
        df: Optional[pd.DataFrame] = payload.get('df')
        return {
            'data': df.to_dict(orient='records') if df is not None else [],
            'colnames': payload.get('colnames'),
            'coltypes': payload.get('coltypes'),
            'rowcount': payload.get('rowcount'),
            'sql_rowcount': payload.get('sql_rowcount')
        }

    @deprecated(deprecated_in='3.0')
    def get_df(self, query_obj: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Returns a pandas dataframe based on the query object"""
        if not query_obj:
            query_obj = self.query_obj()
        if not query_obj:
            return pd.DataFrame()
        self.error_msg = ''
        timestamp_format: Optional[str] = None
        if self.datasource.type == 'table':
            granularity_col: Optional[Column] = self.datasource.get_column(query_obj['granularity'])
            if granularity_col:
                timestamp_format = granularity_col.python_date_format
        self.results: QueryResult = self.datasource.query(query_obj)
        self.applied_filter_columns = self.results.applied_filter_columns or []
        self.rejected_filter_columns = self.results.rejected_filter_columns or []
        self.query = self.results.query
        self.status = self.results.status
        self.errors = self.results.errors
        df: pd.DataFrame = self.results.df
        if not df.empty:
            utils.normalize_dttm_col(
                df=df,
                dttm_cols=tuple([
                    DateColumn.get_legacy_time_column(
                        timestamp_format=timestamp_format,
                        offset=self.datasource.offset,
                        time_shift=self.form_data.get('time_shift')
                    )
                ])
            )
            if self.enforce_numerical_metrics:
                self.df_metrics_to_num(df)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
        return df

    @deprecated(deprecated_in='3.0')
    def df_metrics_to_num(self, df: pd.DataFrame) -> None:
        """Converting metrics to numeric when pandas.read_sql cannot"""
        metrics: List[str] = self.metric_labels
        for col, dtype in df.dtypes.items():
            if dtype.type == np.object_ and col in metrics:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    @deprecated(deprecated_in='3.0')
    def process_query_filters(self) -> None:
        utils.convert_legacy_filters_into_adhoc(self.form_data)
        merge_extra_filters(self.form_data)
        utils.split_adhoc_filters_into_base_filters(self.form_data)

    @staticmethod
    @deprecated(deprecated_in='3.0')
    def dedup_columns(*columns_args: List[Any]) -> List[Any]:
        labels: List[Any] = []
        deduped_columns: List[Any] = []
        for columns in columns_args:
            for column in columns or []:
                label: Any = get_column_name(column)
                if label not in labels:
                    deduped_columns.append(column)
                    labels.append(label)
        return deduped_columns

    @deprecated(deprecated_in='3.0')
    def query_obj(self) -> Dict[str, Any]:
        """Building a query object"""
        self.process_query_filters()
        metrics: List[Any] = self.all_metrics or []
        groupby: List[Any] = self.dedup_columns(self.groupby, self.form_data.get('columns'))
        is_timeseries: bool = self.is_timeseries
        groupby_labels: List[str] = get_column_names(groupby)
        if DTTM_ALIAS in groupby_labels:
            del groupby[groupby_labels.index(DTTM_ALIAS)]
            is_timeseries = True
        granularity: Any = self.form_data.get('granularity_sqla')
        limit: int = int(self.form_data.get('limit') or 0)
        timeseries_limit_metric: Any = self.form_data.get('timeseries_limit_metric')
        row_limit: int = int(self.form_data.get('row_limit') or config['ROW_LIMIT'])
        row_limit = apply_max_row_limit(row_limit)
        order_desc: bool = self.form_data.get('order_desc', True)
        try:
            since: Optional[datetime]
            until: Optional[datetime]
            since, until = get_since_until(
                relative_start=relative_start,
                relative_end=relative_end,
                time_range=self.form_data.get('time_range'),
                since=self.form_data.get('since'),
                until=self.form_data.get('until'),
            )
        except ValueError as ex:
            raise QueryObjectValidationError(str(ex)) from ex
        time_shift: str = self.form_data.get('time_shift', '')
        self.time_shift = parse_past_timedelta(time_shift)
        from_dttm: Optional[datetime] = None if since is None else since - self.time_shift
        to_dttm: Optional[datetime] = None if until is None else until - self.time_shift
        if from_dttm and to_dttm and (from_dttm > to_dttm):
            raise QueryObjectValidationError(_('From date cannot be larger than to date'))
        self.from_dttm = from_dttm
        self.to_dttm = to_dttm
        for param in ('where', 'having'):
            clause: Any = self.form_data.get(param)
            if clause:
                sanitized_clause: str = sanitize_clause(clause)
                if sanitized_clause != clause:
                    self.form_data[param] = sanitized_clause
        extras: Dict[str, Any] = {
            'having': self.form_data.get('having', ''),
            'time_grain_sqla': self.form_data.get('time_grain_sqla'),
            'where': self.form_data.get('where', ''),
        }
        return {
            'granularity': granularity,
            'from_dttm': from_dttm,
            'to_dttm': to_dttm,
            'is_timeseries': is_timeseries,
            'groupby': groupby,
            'metrics': metrics,
            'row_limit': row_limit,
            'filter': self.form_data.get('filters', []),
            'timeseries_limit': limit,
            'extras': extras,
            'timeseries_limit_metric': timeseries_limit_metric,
            'order_desc': order_desc,
        }

    @property
    @deprecated(deprecated_in='3.0')
    def cache_timeout(self) -> int:
        if self.form_data.get('cache_timeout') is not None:
            return int(self.form_data['cache_timeout'])
        if self.datasource.cache_timeout is not None:
            return self.datasource.cache_timeout
        if (
            hasattr(self.datasource, 'database')
            and self.datasource.database.cache_timeout is not None
        ):
            return self.datasource.database.cache_timeout
        if config['DATA_CACHE_CONFIG'].get('CACHE_DEFAULT_TIMEOUT') is not None:
            return config['DATA_CACHE_CONFIG']['CACHE_DEFAULT_TIMEOUT']
        return config['CACHE_DEFAULT_TIMEOUT']

    @deprecated(deprecated_in='3.0')
    def get_json(self) -> str:
        return json.dumps(self.get_payload(), default=json.json_int_dttm_ser, ignore_nan=True)

    @deprecated(deprecated_in='3.0')
    def cache_key(
        self,
        query_obj: Dict[str, Any],
        **extra: Any
    ) -> str:
        """
        The cache key is made out of the key/values in `query_obj`, plus any
        other key/values in `extra`.

        We remove datetime bounds that are hard values, and replace them with
        the use-provided inputs to bounds, which may be time-relative (as in
        "5 days ago" or "now").

        The `extra` arguments are currently used by time shift queries, since
        different time shifts will differ only in the `from_dttm`, `to_dttm`,
        `inner_from_dttm`, and `inner_to_dttm` values which are stripped.
        """
        cache_dict: Dict[str, Any] = copy.copy(query_obj)
        cache_dict.update(extra)
        for k in ['from_dttm', 'to_dttm', 'inner_from_dttm', 'inner_to_dttm']:
            if k in cache_dict:
                del cache_dict[k]
        cache_dict['time_range'] = self.form_data.get('time_range')
        cache_dict['datasource'] = self.datasource.uid
        cache_dict['extra_cache_keys'] = self.datasource.get_extra_cache_keys(query_obj)
        cache_dict['rls'] = security_manager.get_rls_cache_key(self.datasource)
        cache_dict['changed_on'] = self.datasource.changed_on
        json_data: str = self.json_dumps(cache_dict, sort_keys=True)
        return md5_sha_from_str(json_data)

    @deprecated(deprecated_in='3.0')
    def get_payload(self, query_obj: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Returns a payload of metadata and data"""
        try:
            self.run_extra_queries()
        except SupersetSecurityException as ex:
            error: Dict[str, Any] = dataclasses.asdict(ex.error)
            self.errors.append(error)
            self.status = QueryStatus.FAILED
        payload: Dict[str, Any] = self.get_df_payload(query_obj)
        df: Optional[pd.DataFrame] = cast(Optional[pd.DataFrame], payload.get('df'))
        if self.status != QueryStatus.FAILED:
            payload['data'] = self.get_data(df)
        if 'df' in payload:
            del payload['df']
        applied_filter_columns: List[Any] = self.applied_filter_columns or []
        rejected_filter_columns: List[Any] = self.rejected_filter_columns or []
        applied_time_extras: Dict[str, Any] = self.form_data.get('applied_time_extras', {})
        applied_time_columns, rejected_time_columns = utils.get_time_filter_status(
            self.datasource, applied_time_extras
        )
        payload['applied_filters'] = [
            {'column': get_column_name(col)} for col in applied_filter_columns
        ] + applied_time_columns
        payload['rejected_filters'] = [
            {
                'reason': ExtraFiltersReasonType.COL_NOT_IN_DATASOURCE,
                'column': get_column_name(col)
            } for col in rejected_filter_columns
        ] + rejected_time_columns
        if df is not None:
            payload['colnames'] = list(df.columns)
        return payload

    @deprecated(deprecated_in='3.0')
    def get_df_payload(
        self,
        query_obj: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Handles caching around the df payload retrieval"""
        if not query_obj:
            query_obj = self.query_obj()
        cache_key: Optional[str] = self.cache_key(query_obj, **kwargs) if query_obj else None
        cache_value: Optional[Dict[str, Any]] = None
        logger.info('Cache key: %s', cache_key)
        is_loaded: bool = False
        stacktrace: Optional[str] = None
        df: Optional[pd.DataFrame] = None
        cache_timeout: int = self.cache_timeout
        force: bool = self.force or cache_timeout == -1
        if cache_key and cache_manager.data_cache and (not force):
            cache_value = cache_manager.data_cache.get(cache_key)
            if cache_value:
                stats_logger.incr('loading_from_cache')
                try:
                    df = cache_value['df']
                    self.query = cache_value['query']
                    self.applied_filter_columns = cache_value.get('applied_filter_columns', [])
                    self.rejected_filter_columns = cache_value.get('rejected_filter_columns', [])
                    self.status = QueryStatus.SUCCESS
                    is_loaded = True
                    stats_logger.incr('loaded_from_cache')
                except Exception as ex:
                    logger.exception(ex)
                    logger.error(
                        'Error reading cache: %s',
                        utils.error_msg_from_exception(ex),
                        exc_info=True
                    )
                logger.info('Serving from cache')
        if query_obj and (not is_loaded):
            if self.force_cached:
                logger.warning(
                    'force_cached (viz.py): value not found for cache key %s',
                    cache_key
                )
                raise CacheLoadError(_('Cached value not found'))
            try:
                invalid_columns: List[str] = [
                    col for col in (
                        get_column_names_from_columns(query_obj.get('columns') or [])
                        + get_column_names_from_columns(query_obj.get('groupby') or [])
                        + utils.get_column_names_from_metrics(cast(List[Metric], query_obj.get('metrics') or []))
                    )
                    if col not in self.datasource.column_names
                ]
                if invalid_columns:
                    raise QueryObjectValidationError(
                        _('Columns missing in datasource: %(invalid_columns)s', invalid_columns=invalid_columns)
                    )
                df = self.get_df(query_obj)
                if self.status != QueryStatus.FAILED:
                    stats_logger.incr('loaded_from_source')
                    if not self.force:
                        stats_logger.incr('loaded_from_source_without_force')
                    is_loaded = True
            except QueryObjectValidationError as ex:
                error: Dict[str, Any] = dataclasses.asdict(
                    SupersetError(
                        message=str(ex),
                        level=ErrorLevel.ERROR,
                        error_type=SupersetErrorType.VIZ_GET_DF_ERROR,
                    )
                )
                self.errors.append(error)
                self.status = QueryStatus.FAILED
            except Exception as ex:
                logger.exception(ex)
                error: Dict[str, Any] = dataclasses.asdict(
                    SupersetError(
                        message=str(ex),
                        level=ErrorLevel.ERROR,
                        error_type=SupersetErrorType.VIZ_GET_DF_ERROR,
                    )
                )
                self.errors.append(error)
                self.status = QueryStatus.FAILED
                stacktrace = utils.get_stacktrace()
            if is_loaded and cache_key and (self.status != QueryStatus.FAILED):
                set_and_log_cache(
                    cache_instance=cache_manager.data_cache,
                    cache_key=cache_key,
                    cache_value={
                        'df': df,
                        'query': self.query
                    },
                    cache_timeout=cache_timeout,
                    datasource_uid=self.datasource.uid
                )
        return {
            'cache_key': cache_key,
            'cached_dttm': cache_value['dttm'] if cache_value is not None else None,
            'cache_timeout': cache_timeout,
            'df': df,
            'errors': self.errors,
            'form_data': self.form_data,
            'is_cached': cache_value is not None,
            'query': self.query,
            'from_dttm': self.from_dttm,
            'to_dttm': self.to_dttm,
            'status': self.status,
            'stacktrace': stacktrace,
            'rowcount': len(df.index) if df is not None else 0,
            'colnames': list(df.columns) if df is not None else None,
            'coltypes': utils.extract_dataframe_dtypes(df, self.datasource) if df is not None else None
        }

    @staticmethod
    @deprecated(deprecated_in='3.0')
    def json_dumps(query_obj: Dict[str, Any], sort_keys: bool = False) -> str:
        return json.dumps(
            query_obj,
            default=json.json_int_dttm_ser,
            ignore_nan=True,
            sort_keys=sort_keys
        )

    @staticmethod
    @deprecated(deprecated_in='3.0')
    def has_error(payload: Dict[str, Any]) -> bool:
        return (
            payload.get('status') == QueryStatus.FAILED
            or payload.get('error') is not None
            or bool(payload.get('errors'))
        )

    @deprecated(deprecated_in='3.0')
    def payload_json_and_has_error(
        self,
        payload: Dict[str, Any]
    ) -> Tuple[str, bool]:
        return (self.json_dumps(payload), self.has_error(payload))

    @property
    @deprecated(deprecated_in='3.0')
    def data(self) -> Dict[str, Any]:
        """This is the data object serialized to the js layer"""
        content: Dict[str, Any] = {
            'form_data': self.form_data,
            'token': self.token,
            'viz_name': self.viz_type,
            'filter_select_enabled': self.datasource.filter_select_enabled
        }
        return content

    @deprecated(deprecated_in='3.0')
    def get_csv(self) -> str:
        df_payload: Dict[str, Any] = self.get_df_payload()
        df: Optional[pd.DataFrame] = df_payload.get('df')
        include_index: bool = not isinstance(df.index, pd.RangeIndex) if df is not None else False
        return csv.df_to_escaped_csv(
            df,
            index=include_index,
            **config['CSV_EXPORT']
        )

    @deprecated(deprecated_in='3.0')
    def get_data(self, df: Optional[pd.DataFrame]) -> Union[List[Dict[str, Any]], None]:
        return df.to_dict(orient='records') if df is not None else None

    @property
    @deprecated(deprecated_in='3.0')
    def json_data(self) -> str:
        return json.dumps(self.data)

    @deprecated(deprecated_in='3.0')
    def raise_for_access(self) -> None:
        """
        Raise an exception if the user cannot access the resource.

        :raises SupersetSecurityException: If the user cannot access the resource
        """
        security_manager.raise_for_access(viz=self)

class TimeTableViz(BaseViz):
    """A data table with rich time-series related columns"""
    viz_type: str = 'time_table'
    verbose_name: str = _('Time Table View')
    credits: str = 'a <a href="https://github.com/airbnb/superset">Superset</a> original'
    is_timeseries: bool = True

    @deprecated(deprecated_in='3.0')
    def query_obj(self) -> Dict[str, Any]:
        query_obj: Dict[str, Any] = super().query_obj()
        if not self.form_data.get('metrics'):
            raise QueryObjectValidationError(_('Pick at least one metric'))
        if self.form_data.get('groupby') and len(self.form_data['metrics']) > 1:
            raise QueryObjectValidationError(_("When using 'Group By' you are limited to use a single metric"))
        sort_by: Optional[str] = utils.get_first_metric_name(query_obj.get('metrics') or [])
        is_asc: bool = not query_obj.get('order_desc', True)
        if sort_by:
            query_obj['orderby'] = [(sort_by, is_asc)]
        return query_obj

    @deprecated(deprecated_in='3.0')
    def get_data(self, df: Optional[pd.DataFrame]) -> Optional[Dict[str, Any]]:
        if df is None or df.empty:
            return None
        columns: Optional[List[str]] = None
        values: Union[List[str], str] = self.metric_labels
        if self.form_data.get('groupby'):
            values = self.metric_labels[0]
            columns = get_column_names(self.form_data.get('groupby'))
        pt: pd.DataFrame = df.pivot_table(
            index=DTTM_ALIAS,
            columns=columns,
            values=values
        )
        pt.index = pt.index.map(str)
        pt = pt.sort_index()
        return {
            'records': pt.to_dict(orient='index'),
            'columns': list(pt.columns),
            'is_group_by': bool(self.form_data.get('groupby'))
        }

class CalHeatmapViz(BaseViz):
    """Calendar heatmap."""
    viz_type: str = 'cal_heatmap'
    verbose_name: str = _('Calendar Heatmap')
    credits: str = '<a href=https://github.com/wa0x6e/cal-heatmap>cal-heatmap</a>'
    is_timeseries: bool = True

    @deprecated(deprecated_in='3.0')
    def get_data(self, df: Optional[pd.DataFrame]) -> Optional[Dict[str, Any]]:
        if df is None or df.empty:
            return None
        form_data: Dict[str, Any] = self.form_data
        data: Dict[str, Dict[str, Any]] = {}
        records: List[Dict[str, Any]] = df.to_dict('records')
        for metric in self.metric_labels:
            values: Dict[str, Any] = {}
            for query_obj in records:
                v: Any = query_obj[DTTM_ALIAS]
                if hasattr(v, 'value'):
                    v = v.value
                values[str(v / 10 ** 9)] = query_obj.get(metric)
            data[metric] = values
        try:
            start: Optional[datetime]
            end: Optional[datetime]
            start, end = get_since_until(
                relative_start=relative_start,
                relative_end=relative_end,
                time_range=form_data.get('time_range'),
                since=form_data.get('since'),
                until=form_data.get('until'),
            )
        except ValueError as ex:
            raise QueryObjectValidationError(str(ex)) from ex
        if not start or not end:
            raise QueryObjectValidationError('Please provide both time bounds (Since and Until)')
        domain: str = form_data.get('domain_granularity')
        diff_delta: rdelta.relativedelta = rdelta.relativedelta(end, start)
        diff_secs: float = (end - start).total_seconds()
        if domain == 'year':
            range_: int = end.year - start.year + 1
        elif domain == 'month':
            range_ = diff_delta.years * 12 + diff_delta.months + 1
        elif domain == 'week':
            range_ = diff_delta.years * 53 + diff_delta.weeks + 1
        elif domain == 'day':
            range_ = int(diff_secs // (24 * 60 * 60)) + 1
        else:
            range_ = int(diff_secs // (60 * 60)) + 1
        return {
            'data': data,
            'start': start,
            'domain': domain,
            'subdomain': form_data.get('subdomain_granularity'),
            'range': range_
        }

    @deprecated(deprecated_in='3.0')
    def query_obj(self) -> Dict[str, Any]:
        query_obj: Dict[str, Any] = super().query_obj()
        query_obj['metrics'] = self.form_data.get('metrics')
        mapping: Dict[str, str] = {
            'min': 'PT1M',
            'hour': 'PT1H',
            'day': 'P1D',
            'week': 'P1W',
            'month': 'P1M',
            'year': 'P1Y'
        }
        query_obj['extras']['time_grain_sqla'] = mapping.get(
            self.form_data.get('subdomain_granularity', 'min'),
            'PT1M'
        )
        return query_obj

class NVD3Viz(BaseViz):
    """Base class for all nvd3 vizs"""
    credits: str = '<a href="http://nvd3.org/">NVD3.org</a>'
    viz_type: Optional[str] = None
    verbose_name: str = 'Base NVD3 Viz'
    is_timeseries: bool = False

class BubbleViz(NVD3Viz):
    """Based on the NVD3 bubble chart"""
    viz_type: str = 'bubble'
    verbose_name: str = _('Bubble Chart')
    is_timeseries: bool = False

    @deprecated(deprecated_in='3.0')
    def query_obj(self) -> Dict[str, Any]:
        query_obj: Dict[str, Any] = super().query_obj()
        query_obj['groupby'] = [self.form_data.get('entity')]
        if self.form_data.get('series'):
            query_obj['groupby'].append(self.form_data.get('series'))
        query_obj['groupby'] = self.dedup_columns(query_obj['groupby'])
        self.x_metric: Any = self.form_data['x']
        self.y_metric: Any = self.form_data['y']
        self.z_metric: Any = self.form_data['size']
        self.entity: Any = self.form_data.get('entity')
        self.series: Any = self.form_data.get('series') or self.entity
        query_obj['row_limit'] = self.form_data.get('limit')
        query_obj['metrics'] = [self.z_metric, self.x_metric, self.y_metric]
        if len(set(self.metric_labels)) < 3:
            raise QueryObjectValidationError(_('Please use 3 different metric labels'))
        if not all(query_obj['metrics'] + [self.entity]):
            raise QueryObjectValidationError(_('Pick a metric for x, y and size'))
        return query_obj

    @deprecated(deprecated_in='3.0')
    def get_data(self, df: Optional[pd.DataFrame]) -> Optional[List[Dict[str, Any]]]:
        if df is None or df.empty:
            return None
        df['x'] = df[[utils.get_metric_name(self.x_metric)]]
        df['y'] = df[[utils.get_metric_name(self.y_metric)]]
        df['size'] = df[[utils.get_metric_name(self.z_metric)]]
        df['shape'] = 'circle'
        df['group'] = df[[get_column_name(self.series)]]
        series: defaultdict = defaultdict(list)
        for row in df.to_dict(orient='records'):
            series[row['group']].append(row)
        chart_data: List[Dict[str, Any]] = []
        for k, v in series.items():
            chart_data.append({'key': k, 'values': v})
        return chart_data

class BulletViz(NVD3Viz):
    """Based on the NVD3 bullet chart"""
    viz_type: str = 'bullet'
    verbose_name: str = _('Bullet Chart')
    is_timeseries: bool = False

    @deprecated(deprecated_in='3.0')
    def query_obj(self) -> Dict[str, Any]:
        form_data: Dict[str, Any] = self.form_data
        query_obj: Dict[str, Any] = super().query_obj()
        self.metric: Any = form_data['metric']
        query_obj['metrics'] = [self.metric]
        if not self.metric:
            raise QueryObjectValidationError(_('Pick a metric to display'))
        return query_obj

    @deprecated(deprecated_in='3.0')
    def get_data(self, df: Optional[pd.DataFrame]) -> Optional[Dict[str, Any]]:
        if df is None or df.empty:
            return None
        df['metric'] = df[[utils.get_metric_name(self.metric)]]
        values: np.ndarray = df['metric'].values
        return {'measures': values.tolist()}

class NVD3TimeSeriesViz(NVD3Viz):
    """A rich line chart component with tons of options"""
    viz_type: str = 'line'
    verbose_name: str = _('Time Series - Line Chart')
    sort_series: bool = False
    is_timeseries: bool = True
    pivot_fill_value: Any = None

    @deprecated(deprecated_in='3.0')
    def query_obj(self) -> Dict[str, Any]:
        query_obj: Dict[str, Any] = super().query_obj()
        sort_by: Optional[str] = self.form_data.get('timeseries_limit_metric') or utils.get_first_metric_name(query_obj.get('metrics') or [])
        is_asc: bool = not self.form_data.get('order_desc')
        if sort_by:
            sort_by_label: str = utils.get_metric_name(sort_by)
            if sort_by_label not in utils.get_metric_names(query_obj['metrics']):
                query_obj['metrics'].append(sort_by)
            query_obj['orderby'] = [(sort_by, is_asc)]
        return query_obj

    @deprecated(deprecated_in='3.0')
    def to_series(
        self,
        df: pd.DataFrame,
        classed: str = '',
        title_suffix: str = ''
    ) -> List[Dict[str, Any]]:
        cols: List[str] = []
        for col in df.columns:
            if col == '':
                cols.append('N/A')
            elif col is None:
                cols.append('NULL')
            else:
                cols.append(col)
        df.columns = cols
        series: Dict[str, pd.Series] = df.to_dict('series')
        chart_data: List[Dict[str, Any]] = []
        for name in df.T.index.tolist():
            ys: pd.Series = series[name]
            if df[name].dtype.kind not in 'biufc':
                continue
            if isinstance(name, list):
                series_title: Union[str, List[str]] = [str(title) for title in name]
            elif isinstance(name, tuple):
                series_title = tuple((str(title) for title in name))
            else:
                series_title = str(name)
            if isinstance(series_title, (list, tuple)) and len(series_title) > 1 and (len(self.metric_labels) == 1):
                series_title = series_title[1:]
            if title_suffix:
                if isinstance(series_title, str):
                    series_title = (series_title, title_suffix)
                elif isinstance(series_title, list):
                    series_title = series_title + [title_suffix]
                elif isinstance(series_title, tuple):
                    series_title = series_title + (title_suffix,)
            values: List[Dict[str, Any]] = []
            non_nan_cnt: int = 0
            for ds in df.index:
                if ds in ys:
                    data: Dict[str, Any] = {'x': ds, 'y': ys[ds]}
                    if not np.isnan(ys[ds]):
                        non_nan_cnt += 1
                else:
                    data = {}
                values.append(data)
            if non_nan_cnt == 0:
                continue
            data: Dict[str, Any] = {'key': series_title, 'values': values}
            if classed:
                data['classed'] = classed
            chart_data.append(data)
        return chart_data

    @deprecated(deprecated_in='3.0')
    def process_data(
        self,
        df: pd.DataFrame,
        aggregate: bool = False
    ) -> pd.DataFrame:
        if df.empty:
            return df
        if aggregate:
            df = df.pivot_table(
                index=DTTM_ALIAS,
                columns=get_column_names(self.form_data.get('groupby')),
                values=self.metric_labels,
                fill_value=0,
                aggfunc=sum
            )
        else:
            df = df.pivot_table(
                index=DTTM_ALIAS,
                columns=get_column_names(self.form_data.get('groupby')),
                values=self.metric_labels,
                fill_value=self.pivot_fill_value
            )
        rule: Optional[str] = self.form_data.get('resample_rule')
        method: Optional[str] = self.form_data.get('resample_method')
        if rule and method:
            if hasattr(df.resample(rule), method):
                df = getattr(df.resample(rule), method)()
        if self.sort_series:
            dfs: pd.Series = df.sum()
            dfs.sort_values(ascending=False, inplace=True)
            df = df[dfs.index]
        df = self.apply_rolling(df)
        if self.form_data.get('contribution'):
            dft: pd.DataFrame = df.T
            df = (dft / dft.sum()).T
        return df

    @deprecated(deprecated_in='3.0')
    def run_extra_queries(self) -> None:
        time_compare: Union[List[Any], Any] = self.form_data.get('time_compare') or []
        if not isinstance(time_compare, list):
            time_compare = [time_compare]
        for option in time_compare:
            query_object: Dict[str, Any] = self.query_obj()
            try:
                delta: timedelta = parse_past_timedelta(option)
            except ValueError as ex:
                raise QueryObjectValidationError(str(ex)) from ex
            query_object['inner_from_dttm'] = query_object['from_dttm']
            query_object['inner_to_dttm'] = query_object['to_dttm']
            if not query_object['from_dttm'] or not query_object['to_dttm']:
                raise QueryObjectValidationError(_('An enclosed time range (both start and end) must be specified when using a Time Comparison.'))
            query_object['from_dttm'] -= delta
            query_object['to_dttm'] -= delta
            df2: Optional[pd.DataFrame] = self.get_df_payload(
                query_object, time_compare=option
            ).get('df')
            if df2 is not None and DTTM_ALIAS in df2:
                dttm_series: pd.Series = df2[DTTM_ALIAS] + delta
                df2 = df2.drop(DTTM_ALIAS, axis=1)
                df2 = pd.concat([dttm_series, df2], axis=1)
                label: str = f'{option} offset'
                df2 = self.process_data(df2)
                self._extra_chart_data.append((label, df2))

    @deprecated(deprecated_in='3.0')
    def get_data(self, df: Optional[pd.DataFrame]) -> Optional[List[Dict[str, Any]]]:
        comparison_type: str = self.form_data.get('comparison_type') or 'values'
        df = self.process_data(df)
        if comparison_type == 'values':
            chart_data: List[Dict[str, Any]] = self.to_series(df.dropna(axis=1, how='all'))
            for i, (label, df2) in enumerate(self._extra_chart_data):
                chart_data.extend(self.to_series(df2, classed=f'time-shift-{i}', title_suffix=label))
        else:
            chart_data: List[Dict[str, Any]] = []
            for i, (label, df2) in enumerate(self._extra_chart_data):
                combined_index: pd.Index = df.index.union(df2.index)
                df2 = df2.reindex(combined_index).interpolate(method='time').reindex(df.index)
                if comparison_type == 'absolute':
                    diff: pd.DataFrame = df - df2
                elif comparison_type == 'percentage':
                    diff = (df - df2) / df2
                elif comparison_type == 'ratio':
                    diff = df / df2
                else:
                    raise QueryObjectValidationError(f'Invalid `comparison_type`: {comparison_type}')
                diff = diff[diff.first_valid_index():diff.last_valid_index()]
                chart_data.extend(self.to_series(diff, classed=f'time-shift-{i}', title_suffix=label))
        if not self.sort_series:
            chart_data = sorted(chart_data, key=lambda x: tuple(x['key']))
        return chart_data

class NVD3TimePivotViz(NVD3TimeSeriesViz):
    """Time Series - Periodicity Pivot"""
    viz_type: str = 'time_pivot'
    sort_series: bool = True
    verbose_name: str = _('Time Series - Period Pivot')

    @deprecated(deprecated_in='3.0')
    def query_obj(self) -> Dict[str, Any]:
        query_obj: Dict[str, Any] = super().query_obj()
        query_obj['metrics'] = [self.form_data.get('metric')]
        return query_obj

    @deprecated(deprecated_in='3.0')
    def get_data(self, df: Optional[pd.DataFrame]) -> Optional[List[Dict[str, Any]]]:
        if df is None or df.empty:
            return None
        df = self.process_data(df)
        freq = to_offset(self.form_data.get('freq'))
        try:
            freq = type(freq)(freq.n, normalize=True, **freq.kwds)
        except ValueError:
            freq = type(freq)(freq.n, **freq.kwds)
        df.index.name = None
        df[DTTM_ALIAS] = df.index.map(freq.rollback)
        df['ranked'] = df[DTTM_ALIAS].rank(method='dense', ascending=False) - 1
        df.ranked = df.ranked.map(int)
        df['series'] = '-' + df.ranked.map(str)
        df['series'] = df['series'].str.replace('-0', 'current')
        rank_lookup: Dict[str, int] = {
            row['series']: row['ranked'] for row in df.to_dict(orient='records')
        }
        max_ts: datetime = df[DTTM_ALIAS].max()
        max_rank: int = df['ranked'].max()
        df[DTTM_ALIAS] = df.index + (max_ts - df[DTTM_ALIAS])
        df = df.pivot_table(
            index=DTTM_ALIAS,
            columns='series',
            values=utils.get_metric_name(self.form_data['metric'])
        )
        chart_data: List[Dict[str, Any]] = self.to_series(df)
        for series in chart_data:
            series['rank'] = rank_lookup[series['key']]
            series['perc'] = 1 - series['rank'] / (max_rank + 1)
        return chart_data

class NVD3CompareTimeSeriesViz(NVD3TimeSeriesViz):
    """A line chart component where you can compare the % change over time"""
    viz_type: str = 'compare'
    verbose_name: str = _('Time Series - Percent Change')

class ChordViz(BaseViz):
    """A Chord diagram"""
    viz_type: str = 'chord'
    verbose_name: str = _('Directed Force Layout')
    credits: str = '<a href="https://github.com/d3/d3-chord">Bostock</a>'
    is_timeseries: bool = False

    @deprecated(deprecated_in='3.0')
    def query_obj(self) -> Dict[str, Any]:
        query_obj: Dict[str, Any] = super().query_obj()
        query_obj['groupby'] = [
            self.form_data.get('groupby'),
            self.form_data.get('columns')
        ]
        query_obj['metrics'] = [self.form_data.get('metric')]
        if self.form_data.get('sort_by_metric', False):
            query_obj['orderby'] = [(query_obj['metrics'][0], False)]
        return query_obj

    @deprecated(deprecated_in='3.0')
    def get_data(self, df: Optional[pd.DataFrame]) -> Optional[Dict[str, Any]]:
        if df is None or df.empty:
            return None
        df.columns = ['source', 'target', 'value']
        nodes: List[str] = list(set(df['source']) | set(df['target']))
        matrix: Dict[Tuple[str, str], float] = {}
        for source, target in product(nodes, nodes):
            matrix[source, target] = 0
        for source, target, value in df.to_records(index=False):
            matrix[source, target] = value
        return {
            'nodes': list(nodes),
            'matrix': [
                [matrix[n1, n2] for n1 in nodes] for n2 in nodes
            ]
        }

class CountryMapViz(BaseViz):
    """A country centric"""
    viz_type: str = 'country_map'
    verbose_name: str = _('Country Map')
    is_timeseries: bool = False
    credits: str = 'From bl.ocks.org By john-guerra'

    @deprecated(deprecated_in='3.0')
    def query_obj(self) -> Dict[str, Any]:
        query_obj: Dict[str, Any] = super().query_obj()
        metric: Any = self.form_data.get('metric')
        entity: Any = self.form_data.get('entity')
        if not self.form_data.get('select_country'):
            raise QueryObjectValidationError('Must specify a country')
        if not metric:
            raise QueryObjectValidationError('Must specify a metric')
        if not entity:
            raise QueryObjectValidationError('Must provide ISO codes')
        query_obj['metrics'] = [metric]
        query_obj['groupby'] = [entity]
        return query_obj

    @deprecated(deprecated_in='3.0')
    def get_data(self, df: Optional[pd.DataFrame]) -> Optional[List[Dict[str, Any]]]:
        if df is None or df.empty:
            return None
        cols: List[str] = get_column_names([self.form_data.get('entity')])
        metric: str = self.metric_labels[0]
        cols += [metric]
        ndf: pd.DataFrame = df[cols]
        df = ndf
        df.columns = ['country_id', 'metric']
        return df.to_dict(orient='records')

class WorldMapViz(BaseViz):
    """A country centric world map"""
    viz_type: str = 'world_map'
    verbose_name: str = _('World Map')
    is_timeseries: bool = False
    credits: str = 'datamaps on <a href="https://www.npmjs.com/package/datamaps">npm</a>'

    @deprecated(deprecated_in='3.0')
    def query_obj(self) -> Dict[str, Any]:
        query_obj: Dict[str, Any] = super().query_obj()
        query_obj['groupby'] = [self.form_data['entity']]
        if self.form_data.get('sort_by_metric', False):
            query_obj['orderby'] = [(query_obj['metrics'][0], False)]
        return query_obj

    @deprecated(deprecated_in='3.0')
    def get_data(self, df: Optional[pd.DataFrame]) -> Optional[List[Dict[str, Any]]]:
        if df is None or df.empty:
            return None
        from superset.examples import countries
        cols: List[str] = get_column_names([self.form_data.get('entity')])
        metric: str = utils.get_metric_name(self.form_data['metric'])
        secondary_metric: Optional[str] = (
            utils.get_metric_name(self.form_data['secondary_metric'])
            if self.form_data.get('secondary_metric')
            else None
        )
        columns: List[str] = ['country', 'm1', 'm2']
        if metric == secondary_metric:
            ndf: pd.DataFrame = df[cols]
            ndf['m1'] = df[metric]
            ndf['m2'] = ndf['m1']
        else:
            if secondary_metric:
                cols += [metric, secondary_metric]
            else:
                cols += [metric]
                columns = ['country', 'm1']
            ndf: pd.DataFrame = df[cols]
        df = ndf
        df.columns = columns
        data: List[Dict[str, Any]] = df.to_dict(orient='records')
        for row in data:
            country: Optional[Dict[str, Any]] = None
            if isinstance(row['country'], str):
                if 'country_fieldtype' in self.form_data:
                    country = countries.get(self.form_data['country_fieldtype'], row['country'])
            if country:
                row['country'] = country['cca3']
                row['latitude'] = country['lat']
                row['longitude'] = country['lng']
                row['name'] = country['name']
            else:
                row['country'] = 'XXX'
        return data

class ParallelCoordinatesViz(BaseViz):
    """Interactive parallel coordinate implementation

    Uses this amazing javascript library
    https://github.com/syntagmatic/parallel-coordinates
    """
    viz_type: str = 'para'
    verbose_name: str = _('Parallel Coordinates')
    credits: str = '<a href="https://syntagmatic.github.io/parallel-coordinates/">Syntagmatic\'s library</a>'
    is_timeseries: bool = False

    @deprecated(deprecated_in='3.0')
    def query_obj(self) -> Dict[str, Any]:
        query_obj: Dict[str, Any] = super().query_obj()
        query_obj['groupby'] = [self.form_data.get('series')]
        if (sort_by := self.form_data.get('timeseries_limit_metric')):
            sort_by_label: str = utils.get_metric_name(sort_by)
            if sort_by_label not in utils.get_metric_names(query_obj['metrics']):
                query_obj['metrics'].append(sort_by)
            if self.form_data.get('order_desc'):
                query_obj['orderby'] = [(sort_by, not self.form_data.get('order_desc', True))]
        return query_obj

    @deprecated(deprecated_in='3.0')
    def get_data(self, df: Optional[pd.DataFrame]) -> Optional[List[Dict[str, Any]]]:
        return df.to_dict(orient='records') if df is not None else None

class HorizonViz(NVD3TimeSeriesViz):
    """Horizon chart

    https://www.npmjs.com/package/d3-horizon-chart
    """
    viz_type: str = 'horizon'
    verbose_name: str = _('Horizon Charts')
    credits: str = '<a href="https://www.npmjs.com/package/d3-horizon-chart">d3-horizon-chart</a>'

class MapboxViz(BaseViz):
    """Rich maps made with Mapbox"""
    viz_type: str = 'mapbox'
    verbose_name: str = _('Mapbox')
    is_timeseries: bool = False
    credits: str = '<a href=https://www.mapbox.com/mapbox-gl-js/api/>Mapbox GL JS</a>'

    @deprecated(deprecated_in='3.0')
    def query_obj(self) -> Dict[str, Any]:
        query_obj: Dict[str, Any] = super().query_obj()
        label_col: Any = self.form_data.get('mapbox_label')
        if not self.form_data.get('groupby'):
            if self.form_data.get('all_columns_x') is None or self.form_data.get('all_columns_y') is None:
                raise QueryObjectValidationError(_('[Longitude] and [Latitude] must be set'))
            query_obj['columns'] = [
                self.form_data.get('all_columns_x'),
                self.form_data.get('all_columns_y')
            ]
            if label_col and len(label_col) >= 1:
                if label_col[0] == 'count':
                    raise QueryObjectValidationError(
                        _("Must have a [Group By] column to have 'count' as the [Label]")
                    )
                query_obj['columns'].append(label_col[0])
            if self.form_data.get('point_radius') != 'Auto':
                query_obj['columns'].append(self.form_data.get('point_radius'))
            query_obj['columns'] = sorted(set(query_obj['columns']))
        else:
            if label_col and len(label_col) >= 1 and (label_col[0] != 'count') and (label_col[0] not in self.form_data['groupby']):
                raise QueryObjectValidationError(_('Choice of [Label] must be present in [Group By]'))
            if self.form_data.get('point_radius') != 'Auto' and self.form_data.get('point_radius') not in self.form_data['groupby']:
                raise QueryObjectValidationError(_('Choice of [Point Radius] must be present in [Group By]'))
            if (
                self.form_data.get('all_columns_x') not in self.form_data['groupby']
                or self.form_data.get('all_columns_y') not in self.form_data['groupby']
            ):
                raise QueryObjectValidationError(_('[Longitude] and [Latitude] columns must be present in [Group By]'))
        return query_obj

    @deprecated(deprecated_in='3.0')
    def get_data(self, df: Optional[pd.DataFrame]) -> Optional[List[Dict[str, Any]]]:
        if df is None or df.empty:
            return None
        label_col: Any = self.form_data.get('mapbox_label')
        has_custom_metric: bool = label_col is not None and len(label_col) > 0
        metric_col: List[Optional[Any]] = [None] * (len(df.index) if df is not None else 0)
        if has_custom_metric:
            if label_col[0] == self.form_data.get('all_columns_x'):
                metric_col = df[self.form_data.get('all_columns_x')].tolist()
            elif label_col[0] == self.form_data.get('all_columns_y'):
                metric_col = df[self.form_data.get('all_columns_y')].tolist()
            else:
                metric_col = df[label_col[0]].tolist()
        point_radius_col: Union[List[Optional[Any]], pd.Series] = (
            [None] * (len(df.index) if df is not None else 0)
            if self.form_data.get('point_radius') == 'Auto'
            else df[self.form_data.get('point_radius')].tolist()
        )
        geo_precision: int = 10
        geo_json: Dict[str, Any] = {
            'type': 'FeatureCollection',
            'features': [
                {
                    'type': 'Feature',
                    'properties': {'metric': metric, 'radius': point_radius},
                    'geometry': {
                        'type': 'Point',
                        'coordinates': [
                            round(lon, geo_precision),
                            round(lat, geo_precision)
                        ]
                    }
                }
                for lon, lat, metric, point_radius in zip(
                    df[self.form_data.get('all_columns_x')],
                    df[self.form_data.get('all_columns_y')],
                    metric_col,
                    point_radius_col,
                    strict=False
                )
                if isinstance(lon, (float, int)) and isinstance(lat, (float, int))
            ]
        }
        x_series: pd.Series = df[self.form_data.get('all_columns_x')]
        y_series: pd.Series = df[self.form_data.get('all_columns_y')]
        south_west: List[float] = [x_series.min(), y_series.min()]
        north_east: List[float] = [x_series.max(), y_series.max()]
        return {
            'geoJSON': geo_json,
            'hasCustomMetric': has_custom_metric,
            'mapboxApiKey': config['MAPBOX_API_KEY'],
            'mapStyle': self.form_data.get('mapbox_style'),
            'aggregatorName': self.form_data.get('pandas_aggfunc'),
            'clusteringRadius': self.form_data.get('clustering_radius'),
            'pointRadiusUnit': self.form_data.get('point_radius_unit'),
            'globalOpacity': self.form_data.get('global_opacity'),
            'bounds': [south_west, north_east],
            'renderWhileDragging': self.form_data.get('render_while_dragging'),
            'tooltip': self.form_data.get('rich_tooltip'),
            'color': self.form_data.get('mapbox_color')
        }

class DeckGLMultiLayer(BaseViz):
    """Pile on multiple DeckGL layers"""
    viz_type: str = 'deck_multi'
    verbose_name: str = _('Deck.gl - Multiple Layers')
    is_timeseries: bool = False
    credits: str = '<a href="https://uber.github.io/deck.gl/">deck.gl</a>'

    @deprecated(deprecated_in='3.0')
    def query_obj(self) -> Dict[str, Any]:
        return {}

    @deprecated(deprecated_in='3.0')
    def get_data(self, df: Optional[pd.DataFrame]) -> Optional[Dict[str, Any]]:
        from superset import db
        from superset.models.slice import Slice
        slice_ids: List[int] = self.form_data.get('deck_slices') or []
        slices: List[Slice] = db.session.query(Slice).filter(Slice.id.in_(slice_ids)).all()
        return {
            'mapboxApiKey': config['MAPBOX_API_KEY'],
            'slices': [slc.data for slc in slices]
        }

class BaseDeckGLViz(BaseViz):
    """Base class for deck.gl visualizations"""
    is_timeseries: bool = False
    credits: str = '<a href="https://uber.github.io/deck.gl/">deck.gl</a>'
    spatial_control_keys: List[str] = []

    @deprecated(deprecated_in='3.0')
    def get_metrics(self) -> List[Any]:
        self.metric: Optional[Any] = self.form_data.get('size')
        return [self.metric] if self.metric else []

    @deprecated(deprecated_in='3.0')
    def process_spatial_query_obj(self, key: str, group_by: List[Any]) -> None:
        group_by.extend(self.get_spatial_columns(key))

    @deprecated(deprecated_in='3.0')
    def get_spatial_columns(self, key: str) -> List[Any]:
        spatial: Optional[Dict[str, Any]] = self.form_data.get(key)
        if spatial is None:
            raise ValueError(_('Bad spatial key'))
        if spatial.get('type') == 'latlong':
            return [spatial.get('lonCol'), spatial.get('latCol')]
        if spatial.get('type') == 'delimited':
            return [spatial.get('lonlatCol')]
        if spatial.get('type') == 'geohash':
            return [spatial.get('geohashCol')]
        return []

    @staticmethod
    @deprecated(deprecated_in='3.0')
    def parse_coordinates(latlong: Union[str, None]) -> Optional[Tuple[float, float]]:
        if not latlong:
            return None
        try:
            point: Point = Point(latlong)
            return (point.latitude, point.longitude)
        except Exception as ex:
            raise SpatialException(
                _('Invalid spatial point encountered: %(latlong)s', latlong=latlong)
            ) from ex

    @staticmethod
    @deprecated(deprecated_in='3.0')
    def reverse_geohash_decode(geohash_code: str) -> Tuple[float, float]:
        lat, lng = geohash.decode(geohash_code)
        return (lng, lat)

    @staticmethod
    @deprecated(deprecated_in='3.0')
    def reverse_latlong(
        df: pd.DataFrame,
        key: str
    ) -> None:
        df[key] = [tuple(reversed(o)) for o in df[key] if isinstance(o, (list, tuple))]

    @deprecated(deprecated_in='3.0')
    def process_spatial_data_obj(
        self,
        key: str,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        spatial: Optional[Dict[str, Any]] = self.form_data.get(key)
        if spatial is None:
            raise ValueError(_('Bad spatial key'))
        if spatial.get('type') == 'latlong':
            df[key] = list(zip(
                pd.to_numeric(df[spatial.get('lonCol')], errors='coerce'),
                pd.to_numeric(df[spatial.get('latCol')], errors='coerce')
            ))
        elif spatial.get('type') == 'delimited':
            lon_lat_col: str = spatial.get('lonlatCol', '')
            df[key] = df[lon_lat_col].apply(self.parse_coordinates)
            del df[lon_lat_col]
        elif spatial.get('type') == 'geohash':
            geohash_col: str = spatial.get('geohashCol', '')
            df[key] = df[geohash_col].map(self.reverse_geohash_decode)
            del df[geohash_col]
        if spatial.get('reverseCheckbox'):
            self.reverse_latlong(df, key)
        if df.get(key).isnull().all():
            raise NullValueException(_('Encountered invalid NULL spatial entry, please consider filtering those out'))
        return df

    @deprecated(deprecated_in='3.0')
    def add_null_filters(self) -> None:
        spatial_columns: set = set()
        for key in self.spatial_control_keys:
            for column in self.get_spatial_columns(key):
                spatial_columns.add(column)
        if self.form_data.get('adhoc_filters') is None:
            self.form_data['adhoc_filters'] = []
        if (line_column := self.form_data.get('line_column')):
            spatial_columns.add(line_column)
        for column in sorted(spatial_columns):
            filter_: Dict[str, Any] = simple_filter_to_adhoc({
                'col': column,
                'op': 'IS NOT NULL',
                'val': ''
            })
            self.form_data['adhoc_filters'].append(filter_)

    @deprecated(deprecated_in='3.0')
    def query_obj(self) -> Dict[str, Any]:
        if self.form_data.get('filter_nulls', True):
            self.add_null_filters()
        query_obj: Dict[str, Any] = super().query_obj()
        group_by: List[Any] = []
        for key in self.spatial_control_keys:
            self.process_spatial_query_obj(key, group_by)
        if self.form_data.get('dimension'):
            group_by += [self.form_data['dimension']]
        if self.form_data.get('js_columns'):
            group_by += self.form_data.get('js_columns') or []
        group_by = sorted(set(group_by))
        metrics: List[Any] = self.get_metrics()
        if metrics:
            query_obj['groupby'] = group_by
            query_obj['metrics'] = metrics
            query_obj['columns'] = []
            first_metric: str = query_obj['metrics'][0]
            query_obj['orderby'] = [(first_metric, not self.form_data.get('order_desc', True))]
        else:
            query_obj['columns'] = group_by
        return query_obj

    @deprecated(deprecated_in='3.0')
    def get_js_columns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        cols: List[str] = self.form_data.get('js_columns') or []
        return {col: data.get(col) for col in cols}

    @deprecated(deprecated_in='3.0')
    def get_data(self, df: Optional[pd.DataFrame]) -> Optional[Dict[str, Any]]:
        if df is None or df.empty:
            return None
        for key in self.spatial_control_keys:
            df = self.process_spatial_data_obj(key, df)
        features: List[Dict[str, Any]] = []
        for data in df.to_dict(orient='records'):
            feature: Dict[str, Any] = self.get_properties(data)
            extra_props: Dict[str, Any] = self.get_js_columns(data)
            if extra_props:
                feature['extraProps'] = extra_props
            features.append(feature)
        return {
            'features': features,
            'mapboxApiKey': config['MAPBOX_API_KEY'],
            'metricLabels': self.metric_labels
        }

    @deprecated(deprecated_in='3.0')
    def get_properties(self, data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError()

class DeckScatterViz(BaseDeckGLViz):
    """deck.gl's ScatterLayer"""
    viz_type: str = 'deck_scatter'
    verbose_name: str = _('Deck.gl - Scatter plot')
    spatial_control_keys: List[str] = ['spatial']
    is_timeseries: bool = True

    @deprecated(deprecated_in='3.0')
    def query_obj(self) -> Dict[str, Any]:
        self.is_timeseries = bool(self.form_data.get('time_grain_sqla'))
        self.point_radius_fixed: Dict[str, Any] = self.form_data.get('point_radius_fixed') or {'type': 'fix', 'value': 500}
        return super().query_obj()

    @deprecated(deprecated_in='3.0')
    def get_metrics(self) -> List[Any]:
        self.metric = None
        if self.point_radius_fixed.get('type') == 'metric':
            self.metric = self.point_radius_fixed.get('value')
            return [self.metric]
        return []

    @deprecated(deprecated_in='3.0')
    def get_properties(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'metric': data.get(self.metric_label) if self.metric_label else None,
            'radius': self.fixed_value if self.fixed_value else data.get(self.metric_label) if self.metric_label else None,
            'cat_color': data.get(self.dim) if self.dim else None,
            DTTM_ALIAS: data.get(DTTM_ALIAS)
        }

    @deprecated(deprecated_in='3.0')
    def get_data(self, df: Optional[pd.DataFrame]) -> Optional[List[Dict[str, Any]]]:
        self.metric_label: Optional[str] = utils.get_metric_name(self.metric) if self.metric else None
        self.point_radius_fixed: Dict[str, Any] = self.form_data.get('point_radius_fixed')
        self.fixed_value: Optional[float] = None
        self.dim: Optional[str] = self.form_data.get('dimension')
        if self.point_radius_fixed and self.point_radius_fixed.get('type') != 'metric':
            self.fixed_value = self.point_radius_fixed.get('value')
        return super().get_data(df)

class DeckScreengrid(BaseDeckGLViz):
    """deck.gl's ScreenGridLayer"""
    viz_type: str = 'deck_screengrid'
    verbose_name: str = _('Deck.gl - Screen Grid')
    spatial_control_keys: List[str] = ['spatial']
    is_timeseries: bool = True

    @deprecated(deprecated_in='3.0')
    def query_obj(self) -> Dict[str, Any]:
        self.is_timeseries = bool(self.form_data.get('time_grain_sqla'))
        return super().query_obj()

    @deprecated(deprecated_in='3.0')
    def get_properties(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'position': data.get('spatial'),
            'weight': (data.get(self.metric_label) if self.metric_label else None) or 1,
            '__timestamp': data.get(DTTM_ALIAS) or data.get('__time')
        }

    @deprecated(deprecated_in='3.0')
    def get_data(self, df: Optional[pd.DataFrame]) -> Optional[Dict[str, Any]]:
        self.metric_label: Optional[str] = utils.get_metric_name(self.metric) if self.metric else None
        return super().get_data(df)

class DeckGrid(BaseDeckGLViz):
    """deck.gl's DeckLayer"""
    viz_type: str = 'deck_grid'
    verbose_name: str = _('Deck.gl - 3D Grid')
    spatial_control_keys: List[str] = ['spatial']

    @deprecated(deprecated_in='3.0')
    def get_properties(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'position': data.get('spatial'),
            'weight': (data.get(self.metric_label) if self.metric_label else None) or 1
        }

    @deprecated(deprecated_in='3.0')
    def get_data(self, df: Optional[pd.DataFrame]) -> Optional[Dict[str, Any]]:
        self.metric_label: Optional[str] = utils.get_metric_name(self.metric) if self.metric else None
        return super().get_data(df)

@deprecated(deprecated_in='3.0')
def geohash_to_json(geohash_code: str) -> List[List[float]]:
    bbox: Dict[str, float] = geohash.bbox(geohash_code)
    return [
        [bbox.get('w'), bbox.get('n')],
        [bbox.get('e'), bbox.get('n')],
        [bbox.get('e'), bbox.get('s')],
        [bbox.get('w'), bbox.get('s')],
        [bbox.get('w'), bbox.get('n')]
    ]

class DeckPathViz(BaseDeckGLViz):
    """deck.gl's PathLayer"""
    viz_type: str = 'deck_path'
    verbose_name: str = _('Deck.gl - Paths')
    deck_viz_key: str = 'path'
    is_timeseries: bool = True
    deser_map: Dict[str, Any] = {
        'json': json.loads,
        'polyline': polyline.decode,
        'geohash': geohash_to_json
    }

    @deprecated(deprecated_in='3.0')
    def query_obj(self) -> Dict[str, Any]:
        self.metric_label: Optional[str] = utils.get_metric_name(self.metric) if self.metric else None
        self.metric: Optional[Any] = self.form_data.get('metric')
        line_col: Any = self.form_data.get('line_column')
        if self.metric:
            self.has_metrics: bool = True
            self.form_data['groupby'].append(line_col)
        else:
            self.has_metrics: bool = False
            self.form_data['columns'].append(line_col)
        return super().query_obj()

    @deprecated(deprecated_in='3.0')
    def get_properties(self, data: Dict[str, Any]) -> Dict[str, Any]:
        line_type: str = self.form_data['line_type']
        deser = self.deser_map.get(line_type)
        if not deser:
            raise ValueError(_('Invalid line_type'))
        line_column: str = self.form_data['line_column']
        path: Any = deser(data[line_column])
        if self.form_data.get('reverse_long_lat'):
            path = [(o[1], o[0]) for o in path if isinstance(o, (list, tuple))]
        data[self.deck_viz_key] = path
        if line_type != 'geohash':
            del data[line_column]
        data['__timestamp'] = data.get(DTTM_ALIAS) or data.get('__time')
        return data

    @deprecated(deprecated_in='3.0')
    def get_data(self, df: Optional[pd.DataFrame]) -> Optional[Dict[str, Any]]:
        self.metric_label: Optional[str] = utils.get_metric_name(self.metric) if self.metric else None
        return super().get_data(df)

class DeckPolygon(DeckPathViz):
    """deck.gl's Polygon Layer"""
    viz_type: str = 'deck_polygon'
    deck_viz_key: str = 'polygon'
    verbose_name: str = _('Deck.gl - Polygon')

    @deprecated(deprecated_in='3.0')
    def query_obj(self) -> Dict[str, Any]:
        self.elevation: Dict[str, Any] = self.form_data.get('point_radius_fixed') or {'type': 'fix', 'value': 500}
        return super().query_obj()

    @deprecated(deprecated_in='3.0')
    def get_metrics(self) -> List[Any]:
        metrics: List[Any] = [self.form_data.get('metric')]
        if self.elevation.get('type') == 'metric':
            metrics.append(self.elevation.get('value'))
        return [metric for metric in metrics if metric]

    @deprecated(deprecated_in='3.0')
    def get_properties(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = super().get_properties(data)
        elevation: Any = self.form_data['point_radius_fixed']['value']
        type_: str = self.form_data['point_radius_fixed']['type']
        data['elevation'] = data.get(utils.get_metric_name(elevation)) if type_ == 'metric' else elevation
        return data

class DeckHex(BaseDeckGLViz):
    """deck.gl's DeckLayer"""
    viz_type: str = 'deck_hex'
    verbose_name: str = _('Deck.gl - 3D HEX')
    spatial_control_keys: List[str] = ['spatial']

    @deprecated(deprecated_in='3.0')
    def get_properties(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'position': data.get('spatial'),
            'weight': (data.get(self.metric_label) if self.metric_label else None) or 1
        }

    @deprecated(deprecated_in='3.0')
    def get_data(self, df: Optional[pd.DataFrame]) -> Optional[Dict[str, Any]]:
        self.metric_label: Optional[str] = utils.get_metric_name(self.metric) if self.metric else None
        return super().get_data(df)

class DeckHeatmap(BaseDeckGLViz):
    """deck.gl's HeatmapLayer"""
    viz_type: str = 'deck_heatmap'
    verbose_name: str = _('Deck.gl - Heatmap')
    spatial_control_keys: List[str] = ['spatial']

    def get_properties(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'position': data.get('spatial'),
            'weight': (data.get(self.metric_label) if self.metric_label else None) or 1
        }

    def get_data(self, df: Optional[pd.DataFrame]) -> Optional[Dict[str, Any]]:
        self.metric_label: Optional[str] = utils.get_metric_name(self.metric) if self.metric else None
        return super().get_data(df)

class DeckContour(BaseDeckGLViz):
    """deck.gl's ContourLayer"""
    viz_type: str = 'deck_contour'
    verbose_name: str = _('Deck.gl - Contour')
    spatial_control_keys: List[str] = ['spatial']

    def get_properties(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'position': data.get('spatial'),
            'weight': (data.get(self.metric_label) if self.metric_label else None) or 1
        }

    def get_data(self, df: Optional[pd.DataFrame]) -> Optional[Dict[str, Any]]:
        self.metric_label: Optional[str] = utils.get_metric_name(self.metric) if self.metric else None
        return super().get_data(df)

class DeckGeoJson(BaseDeckGLViz):
    """deck.gl's GeoJSONLayer"""
    viz_type: str = 'deck_geojson'
    verbose_name: str = _('Deck.gl - GeoJSON')

    @deprecated(deprecated_in='3.0')
    def query_obj(self) -> Dict[str, Any]:
        query_obj: Dict[str, Any] = super().query_obj()
        query_obj['columns'] += [self.form_data.get('geojson')]
        query_obj['metrics'] = []
        query_obj['groupby'] = []
        return query_obj

    @deprecated(deprecated_in='3.0')
    def get_properties(self, data: Dict[str, Any]) -> Dict[str, Any]:
        geojson: str = data[get_column_name(self.form_data['geojson'])]
        return json.loads(geojson)

class DeckArc(BaseDeckGLViz):
    """deck.gl's Arc Layer"""
    viz_type: str = 'deck_arc'
    verbose_name: str = _('Deck.gl - Arc')
    spatial_control_keys: List[str] = ['start_spatial', 'end_spatial']
    is_timeseries: bool = True

    @deprecated(deprecated_in='3.0')
    def query_obj(self) -> Dict[str, Any]:
        self.is_timeseries = bool(self.form_data.get('time_grain_sqla'))
        return super().query_obj()

    @deprecated(deprecated_in='3.0')
    def get_properties(self, data: Dict[str, Any]) -> Dict[str, Any]:
        dim: Optional[str] = self.form_data.get('dimension')
        return {
            'sourcePosition': data.get('start_spatial'),
            'targetPosition': data.get('end_spatial'),
            'cat_color': data.get(dim) if dim else None,
            DTTM_ALIAS: data.get(DTTM_ALIAS)
        }

    @deprecated(deprecated_in='3.0')
    def get_data(self, df: Optional[pd.DataFrame]) -> Optional[Dict[str, Any]]:
        if df is None or df.empty:
            return None
        return {
            'features': super().get_data(df)['features'],
            'mapboxApiKey': config['MAPBOX_API_KEY']
        }

class EventFlowViz(BaseViz):
    """A visualization to explore patterns in event sequences"""
    viz_type: str = 'event_flow'
    verbose_name: str = _('Event flow')
    credits: str = 'from <a href="https://github.com/williaster/data-ui">@data-ui</a>'
    is_timeseries: bool = True

    @deprecated(deprecated_in='3.0')
    def query_obj(self) -> Dict[str, Any]:
        query: Dict[str, Any] = super().query_obj()
        form_data: Dict[str, Any] = self.form_data
        event_key: Any = form_data['all_columns_x']
        entity_key: Any = form_data['entity']
        meta_keys: List[Any] = [
            col for col in form_data['all_columns'] or []
            if col not in (event_key, entity_key)
        ]
        query['columns'] = [event_key, entity_key] + meta_keys
        if form_data['order_by_entity']:
            query['orderby'] = [(entity_key, True)]
        return query

    @deprecated(deprecated_in='3.0')
    def get_data(self, df: Optional[pd.DataFrame]) -> Optional[List[Dict[str, Any]]]:
        return df.to_dict(orient='records') if df is not None else None

class PairedTTestViz(BaseViz):
    """A table displaying paired t-test values"""
    viz_type: str = 'paired_ttest'
    verbose_name: str = _('Time Series - Paired t-test')
    sort_series: bool = False
    is_timeseries: bool = True

    @deprecated(deprecated_in='3.0')
    def query_obj(self) -> Dict[str, Any]:
        query_obj: Dict[str, Any] = super().query_obj()
        if (sort_by := self.form_data.get('timeseries_limit_metric')):
            sort_by_label: str = utils.get_metric_name(sort_by)
            if sort_by_label not in utils.get_metric_names(query_obj['metrics']):
                query_obj['metrics'].append(sort_by)
            if self.form_data.get('order_desc'):
                query_obj['orderby'] = [(sort_by, not self.form_data.get('order_desc', True))]
        return query_obj

    @deprecated(deprecated_in='3.0')
    def get_data(self, df: Optional[pd.DataFrame]) -> Optional[Dict[str, Any]]:
        if df is None or df.empty:
            return None
        groups: List[str] = get_column_names(self.form_data.get('groupby'))
        time_op: str = self.form_data.get('time_series_option', 'not_time')
        if not groups:
            raise ValueError(_('Please choose at least one groupby'))
        if time_op == 'not_time':
            levels: Dict[int, pd.Series] = self.levels_for(
                'agg_sum',
                groups,
                df
            )
        elif time_op in ['agg_sum', 'agg_mean']:
            levels = self.levels_for(time_op, groups, df)
        elif time_op in ['point_diff', 'point_factor', 'point_percent']:
            levels = self.levels_for_diff(time_op, groups, df)
        elif time_op == 'adv_anal':
            procs: Dict[int, pd.DataFrame] = self.levels_for_time(groups, df)
            return self.nest_procs(procs)
        else:
            levels = self.levels_for('agg_sum', [DTTM_ALIAS] + groups, df)
        return self.nest_values(levels)

    @staticmethod
    @deprecated(deprecated_in='3.0')
    def levels_for(
        time_op: str,
        groups: List[str],
        df: pd.DataFrame
    ) -> Dict[int, pd.Series]:
        """
        Compute the partition at each `level` from the dataframe.
        """
        levels: Dict[int, pd.Series] = {}
        for i in range(0, len(groups) + 1):
            agg_df: pd.DataFrame = df.groupby(groups[:i]) if i else df
            if time_op == 'agg_mean':
                levels[i] = agg_df.mean(numeric_only=True)
            else:
                levels[i] = agg_df.sum(numeric_only=True)
        return levels

    @staticmethod
    @deprecated(deprecated_in='3.0')
    def levels_for_diff(
        time_op: str,
        groups: List[str],
        df: pd.DataFrame
    ) -> Dict[int, pd.Series]:
        times: List[datetime] = list(set(df[DTTM_ALIAS]))
        times.sort()
        until: datetime = times[-1]
        since: datetime = times[0]
        func = {
            'point_diff': [
                pd.Series.sub,
                lambda a, b, fill_value: a - b
            ],
            'point_factor': [
                pd.Series.div,
                lambda a, b, fill_value: a / float(b)
            ],
            'point_percent': [
                lambda a, b, fill_value=0: a.div(b, fill_value=fill_value) - 1,
                lambda a, b, fill_value: a / float(b) - 1
            ]
        }[time_op]
        agg_df: pd.DataFrame = df.groupby(DTTM_ALIAS).sum(numeric_only=True)
        levels: Dict[int, pd.Series] = {
            0: pd.Series({
                m: func[1](agg_df[m][until], agg_df[m][since], 0) for m in agg_df.columns
            })
        }
        for i in range(1, len(groups) + 1):
            agg_df = df.groupby([DTTM_ALIAS] + groups[:i]).sum(numeric_only=True)
            levels[i] = pd.DataFrame({
                m: func[0](agg_df[m][until], agg_df[m][since], fill_value=0) for m in agg_df.columns
            })
        return levels

    @staticmethod
    @deprecated(deprecated_in='3.0')
    def levels_for_time(groups: List[str], df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        procs: Dict[int, pd.DataFrame] = {}
        for i in range(0, len(groups) + 1):
            self_form_data = self.form_data  # type: ignore
            self_form_data['groupby'] = groups[:i]
            df_drop: pd.DataFrame = df.drop(groups[i:], axis=1)
            procs[i] = self.process_data(df_drop, aggregate=True)  # type: ignore
        self.form_data['groupby'] = groups  # type: ignore
        return procs

    @deprecated(deprecated_in='3.0')
    def nest_values(
        self,
        levels: Dict[int, pd.Series],
        level: int = 0,
        metric: Optional[str] = None,
        dims: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Nest values at each level on the back-end with
        access and setting, instead of summing from the bottom.
        """
        if dims is None:
            dims = []
        if not level:
            return [
                {
                    'name': m,
                    'val': levels[0][m],
                    'children': self.nest_values(levels, 1, m)
                } for m in levels[0].index
            ]
        if level == 1:
            metric_level: pd.Series = levels[1][metric]
            return [
                {
                    'name': i,
                    'val': metric_level[i],
                    'children': self.nest_values(levels, 2, metric, [i])
                } for i in metric_level.index
            ]
        if level >= len(levels):
            return []
        dim_level: pd.DataFrame = levels[level][metric][dims[0]]
        return [
            {
                'name': i,
                'val': dim_level[i],
                'children': self.nest_values(levels, level + 1, metric, dims + [i])
            } for i in dim_level.index
        ]

    @deprecated(deprecated_in='3.0')
    def nest_procs(
        self,
        procs: Dict[int, pd.DataFrame],
        level: int = -1,
        dims: Tuple[str, ...] = (),
        time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        if dims is None:
            dims = ()
        if level == -1:
            return [
                {
                    'name': m,
                    'children': self.nest_procs(procs, 0, (m,))
                } for m in procs[0].columns
            ]
        if not level:
            return [
                {
                    'name': t,
                    'val': procs[level][dims[0]][t],
                    'children': self.nest_procs(procs, 1, dims, t)
                } for t in procs[level].index
            ]
        if level >= len(procs):
            return []
        return [
            {
                'name': i,
                'val': procs[level][dims][i][time],
                'children': self.nest_procs(procs, level + 1, dims + (i,), time)
            } for i in procs[level][dims].columns
        ]

    @deprecated(deprecated_in='3.0')
    def get_data(self, df: Optional[pd.DataFrame]) -> Optional[Dict[str, Any]]:
        if df is None or df.empty:
            return None
        groups: List[str] = get_column_names(self.form_data.get('groupby'))
        time_op: str = self.form_data.get('time_series_option', 'not_time')
        if not groups:
            raise ValueError(_('Please choose at least one groupby'))
        if time_op == 'not_time':
            levels: Dict[int, pd.Series] = self.levels_for('agg_sum', groups, df)
        elif time_op in ['agg_sum', 'agg_mean']:
            levels = self.levels_for(time_op, groups, df)
        elif time_op in ['point_diff', 'point_factor', 'point_percent']:
            levels = self.levels_for_diff(time_op, groups, df)
        elif time_op == 'adv_anal':
            procs: Dict[int, pd.DataFrame] = self.levels_for_time(groups, df)
            return self.nest_procs(procs)
        else:
            levels = self.levels_for('agg_sum', [DTTM_ALIAS] + groups, df)
        return self.nest_values(levels)

@deprecated(deprecated_in='3.0')
def get_subclasses(cls: type) -> set:
    return set(cls.__subclasses__()).union([
        sc for c in cls.__subclasses__() for sc in get_subclasses(c)
    ])

viz_types: Dict[str, type(BaseViz)] = {
    o.viz_type: o
    for o in get_subclasses(BaseViz)
    if o.viz_type not in config['VIZ_TYPE_DENYLIST']
}
