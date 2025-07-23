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
from typing import Any, cast, Optional, TYPE_CHECKING, Dict, List, Tuple, Set, Union, Callable, DefaultDict
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
from superset.exceptions import CacheLoadError, NullValueException, QueryObjectValidationError, SpatialException, SupersetSecurityException
from superset.extensions import cache_manager, security_manager
from superset.models.helpers import QueryResult
from superset.sql_parse import sanitize_clause
from superset.superset_typing import Column, Metric, QueryObjectDict, VizData, VizPayload
from superset.utils import core as utils, csv, json
from superset.utils.cache import set_and_log_cache
from superset.utils.core import apply_max_row_limit, DateColumn, DTTM_ALIAS, ExtraFiltersReasonType, get_column_name, get_column_names, get_column_names_from_columns, JS_MAX_INTEGER, merge_extra_filters, simple_filter_to_adhoc
from superset.utils.date_parser import get_since_until, parse_past_timedelta
from superset.utils.hashing import md5_sha_from_str
if TYPE_CHECKING:
    from superset.connectors.sqla.models import BaseDatasource

config = app.config
stats_logger = config['STATS_LOGGER']
relative_start = config['DEFAULT_RELATIVE_START_TIME']
relative_end = config['DEFAULT_RELATIVE_END_TIME']
logger = logging.getLogger(__name__)
METRIC_KEYS = ['metric', 'metrics', 'percent_metrics', 'metric_2', 'secondary_metric', 'x', 'y', 'size']

class BaseViz:
    """All visualizations derive this base class"""
    viz_type: Optional[str] = None
    verbose_name: str = 'Base Viz'
    credits: str = ''
    is_timeseries: bool = False
    cache_type: str = 'df'
    enforce_numerical_metrics: bool = True

    @deprecated(deprecated_in='3.0')
    def __init__(self, datasource: Any, form_data: Dict[str, Any], force: bool = False, force_cached: bool = False) -> None:
        if not datasource:
            raise QueryObjectValidationError(_('Viz is missing a datasource'))
        self.datasource = datasource
        self.request = request
        self.viz_type = form_data.get('viz_type')
        self.form_data = form_data
        self.query = ''
        self.token = utils.get_form_data_token(form_data)
        self.groupby = self.form_data.get('groupby') or []
        self.time_shift = timedelta()
        self.status = None
        self.error_msg = ''
        self.results = None
        self.applied_filter_columns: List[Any] = []
        self.rejected_filter_columns: List[Any] = []
        self.errors: List[Dict[str, Any]] = []
        self.force = force
        self._force_cached = force_cached
        self.from_dttm: Optional[datetime] = None
        self.to_dttm: Optional[datetime] = None
        self._extra_chart_data: List[Tuple[str, pd.DataFrame]] = []
        self.process_metrics()
        self.applied_filters: List[Dict[str, Any]] = []
        self.rejected_filters: List[Dict[str, Any]] = []

    @property
    @deprecated(deprecated_in='3.0')
    def force_cached(self) -> bool:
        return self._force_cached

    @deprecated(deprecated_in='3.0')
    def process_metrics(self) -> None:
        self.metric_dict = OrderedDict()
        for mkey in METRIC_KEYS:
            val = self.form_data.get(mkey)
            if val:
                if not isinstance(val, list):
                    val = [val]
                for o in val:
                    label = utils.get_metric_name(o)
                    self.metric_dict[label] = o
        self.all_metrics = list(self.metric_dict.values())
        self.metric_labels = list(self.metric_dict.keys())

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
        """Lifecycle method to use when more than one query is needed"""
        pass

    @deprecated(deprecated_in='3.0')
    def apply_rolling(self, df: pd.DataFrame) -> pd.DataFrame:
        rolling_type = self.form_data.get('rolling_type')
        rolling_periods = int(self.form_data.get('rolling_periods') or 0)
        min_periods = int(self.form_data.get('min_periods') or 0)
        if rolling_type in ('mean', 'std', 'sum') and rolling_periods:
            kwargs = {'window': rolling_periods, 'min_periods': min_periods}
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
            raise QueryObjectValidationError(_('Applied rolling window did not return any data. Please make sure the source query satisfies the minimum periods defined in the rolling window.'))
        return df

    @deprecated(deprecated_in='3.0')
    def get_samples(self) -> Dict[str, Any]:
        query_obj = self.query_obj()
        query_obj.update({'is_timeseries': False, 'groupby': [], 'metrics': [], 'orderby': [], 'row_limit': config['SAMPLES_ROW_LIMIT'], 'columns': [o.column_name for o in self.datasource.columns], 'from_dttm': None, 'to_dttm': None})
        payload = self.get_df_payload(query_obj)
        return {'data': payload['df'].to_dict(orient='records'), 'colnames': payload.get('colnames'), 'coltypes': payload.get('coltypes'), 'rowcount': payload.get('rowcount'), 'sql_rowcount': payload.get('sql_rowcount')}

    @deprecated(deprecated_in='3.0')
    def get_df(self, query_obj: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Returns a pandas dataframe based on the query object"""
        if not query_obj:
            query_obj = self.query_obj()
        if not query_obj:
            return pd.DataFrame()
        self.error_msg = ''
        timestamp_format = None
        if self.datasource.type == 'table':
            granularity_col = self.datasource.get_column(query_obj['granularity'])
            if granularity_col:
                timestamp_format = granularity_col.python_date_format
        self.results = self.datasource.query(query_obj)
        self.applied_filter_columns = self.results.applied_filter_columns or []
        self.rejected_filter_columns = self.results.rejected_filter_columns or []
        self.query = self.results.query
        self.status = self.results.status
        self.errors = self.results.errors
        df = self.results.df
        if not df.empty:
            utils.normalize_dttm_col(df=df, dttm_cols=tuple([DateColumn.get_legacy_time_column(timestamp_format=timestamp_format, offset=self.datasource.offset, time_shift=self.form_data.get('time_shift'))]))
            if self.enforce_numerical_metrics:
                self.df_metrics_to_num(df)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
        return df

    @deprecated(deprecated_in='3.0')
    def df_metrics_to_num(self, df: pd.DataFrame) -> None:
        """Converting metrics to numeric when pandas.read_sql cannot"""
        metrics = self.metric_labels
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
        labels = []
        deduped_columns = []
        for columns in columns_args:
            for column in columns or []:
                label = get_column_name(column)
                if label not in labels:
                    deduped_columns.append(column)
        return deduped_columns

    @deprecated(deprecated_in='3.0')
    def query_obj(self) -> Dict[str, Any]:
        """Building a query object"""
        self.process_query_filters()
        metrics = self.all_metrics or []
        groupby = self.dedup_columns(self.groupby, self.form_data.get('columns'))
        is_timeseries = self.is_timeseries
        if DTTM_ALIAS in (groupby_labels := get_column_names(groupby)):
            del groupby[groupby_labels.index(DTTM_ALIAS)]
            is_timeseries = True
        granularity = self.form_data.get('granularity_sqla')
        limit = int(self.form_data.get('limit') or 0)
        timeseries_limit_metric = self.form_data.get('timeseries_limit_metric')
        row_limit = int(self.form_data.get('row_limit') or config['ROW_LIMIT'])
        row_limit = apply_max_row_limit(row_limit)
        order_desc = self.form_data.get('order_desc', True)
        try:
            since, until = get_since_until(relative_start=relative_start, relative_end=relative_end, time_range=self.form_data.get('time_range'), since=self.form_data.get('since'), until=self.form_data.get('until'))
        except ValueError as ex:
            raise QueryObjectValidationError(str(ex)) from ex
        time_shift = self.form_data.get('time_shift', '')
        self.time_shift = parse_past_timedelta(time_shift)
        from_dttm = None if since is None else since - self.time_shift
        to_dttm = None if until is None else until - self.time_shift
        if from_dttm and to_dttm and (from_dttm > to_dttm):
            raise QueryObjectValidationError(_('From date cannot be larger than to date'))
        self.from_dttm = from_dttm
        self.to_dttm = to_dttm
        for param in ('where', 'having'):
            clause = self.form_data.get(param)
            if clause:
                sanitized_clause = sanitize_clause(clause)
                if sanitized_clause != clause:
                    self.form_data[param] = sanitized_clause
        extras = {'having': self.form_data.get('having', ''), 'time_grain_sqla': self.form_data.get('time_grain_sqla'), 'where': self.form_data.get('where', '')}
        return {'granularity': granularity, 'from_dttm': from_dttm, 'to_dttm': to_dttm, 'is_timeseries': is_timeseries, 'groupby': groupby, 'metrics': metrics, 'row_limit': row_limit, 'filter': self.form_data.get('filters', []), 'timeseries_limit': limit, 'extras': extras, 'timeseries_limit_metric': timeseries_limit_metric, 'order_desc': order_desc}

    @property
    @deprecated(deprecated_in='3.0')
    def cache_timeout(self) -> int:
        if self.form_data.get('cache_timeout') is not None:
            return int(self.form_data['cache_timeout'])
        if self.datasource.cache_timeout is not None:
            return self.datasource.cache_timeout
        if (hasattr(self.datasource, 'database') and self.datasource.database.cache_timeout) is not None:
            return self.datasource.database.cache_timeout
        if config['DATA_CACHE_CONFIG'].get('CACHE_DEFAULT_TIMEOUT') is not None:
            return config['DATA_CACHE_CONFIG']['CACHE_DEFAULT_TIMEOUT']
        return config['CACHE_DEFAULT_TIMEOUT']

    @deprecated(deprecated_in='3.0')
    def get_json(self) -> str:
        return json.dumps(self.get_payload(), default=json.json_int_dttm_ser, ignore_nan=True)

    @deprecated(deprecated_in='3.0')
    def cache_key(self, query_obj: Dict[str, Any], **extra: Any) -> str:
        """
        The cache key is made out of the key/values in `query_obj`, plus any
        other key/values in `extra`.
        """
        cache_dict = copy.copy(query_obj)
        cache_dict.update(extra)
        for k in ['from_dttm', 'to_dttm', 'inner_from_dttm', 'inner_to_dttm']:
            if k in cache_dict:
                del cache_dict[k]
        cache_dict['time_range'] = self.form_data.get('time_range')
        cache_dict['datasource'] = self.datasource.uid
        cache_dict['extra_cache_keys'] = self.datasource.get_extra_cache_keys(query_obj)
        cache_dict['rls'] = security_manager.get_rls_cache_key(self.datasource)
        cache_dict['changed_on'] = self.datasource.changed_on
        json_data = self.json_dumps(cache_dict, sort_keys=True)
        return md5_sha_from_str(json_data)

    @deprecated(deprecated_in='3.0')
    def get_payload(self, query_obj: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Returns a payload of metadata and data"""
        try:
            self.run_extra_queries()
        except SupersetSecurityException as ex:
            error = dataclasses.asdict(ex.error)
            self.errors.append(error)
            self.status = QueryStatus.FAILED
        payload = self.get_df_payload(query_obj)
        df = cast(Optional[pd.DataFrame], payload['df'])
        if self.status != QueryStatus.FAILED:
            payload['data'] = self.get_data(df)
        if 'df' in payload:
            del payload['df']
        applied_filter_columns = self.applied_filter_columns or []
        rejected_filter_columns = self.rejected_filter_columns or []
        applied_time_extras = self.form_data.get('applied_time_extras', {})
        applied_time_columns, rejected_time_columns = utils.get_time_filter_status(self.datasource, applied_time_extras)
        payload['applied_filters'] = [{'column': get_column_name(col)} for col in applied_filter_columns] + applied_time_columns
        payload['rejected_filters'] = [{'reason': ExtraFiltersReasonType.COL_NOT_IN_DATASOURCE, 'column': get_column_name(col)} for col in rejected_filter_columns] + rejected_time_columns
        if df is not None:
            payload['colnames'] = list(df.columns)
        return payload

    @deprecated(deprecated_in='3.0')
    def get_df_payload(self, query_obj: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
        """Handles caching around the df payload retrieval"""
        if not query_obj:
            query_obj = self.query_obj()
        cache_key = self.cache_key(query_obj, **kwargs) if query_obj else None
        cache_value = None
        logger.info('Cache key: %s', cache_key)
        is_loaded = False
        stacktrace = None
        df = None
        cache_timeout = self.cache_timeout
        force = self.force or cache_timeout == -1
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
                    logger.error('Error reading cache: %s', utils.error_msg_from_exception(ex), exc_info=True)
                logger.info('Serving from cache')
        if query_obj and (not is_loaded):
            if self.force_cached:
                logger.warning('force_cached (viz.py): value not found for cache key %s', cache_key)
                raise CacheLoadError(_('Cached value not found'))
            try:
                invalid_columns = [col for col in get_column_names_from_columns(query_obj.get('columns') or []) + get_column_names_from_columns(query_obj.get('groupby') or []) +