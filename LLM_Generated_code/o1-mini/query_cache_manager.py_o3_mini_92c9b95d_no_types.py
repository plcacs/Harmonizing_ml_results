from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Union
from flask_caching import Cache
from pandas import DataFrame
from superset import app
from superset.common.db_query_status import QueryStatus
from superset.constants import CacheRegion
from superset.exceptions import CacheLoadError
from superset.extensions import cache_manager
from superset.models.helpers import QueryResult
from superset.stats_logger import BaseStatsLogger
from superset.superset_typing import Column
from superset.utils.cache import set_and_log_cache
from superset.utils.core import error_msg_from_exception, get_stacktrace
config = app.config
stats_logger: BaseStatsLogger = config['STATS_LOGGER']
logger: logging.Logger = logging.getLogger(__name__)
_cache: Dict[CacheRegion, Cache] = {CacheRegion.DEFAULT: cache_manager.
    cache, CacheRegion.DATA: cache_manager.data_cache}


class QueryCacheManager:
    """
    Class for manage query-cache getting and setting
    """
    df: DataFrame
    query: str
    annotation_data: Dict[str, Any]
    applied_template_filters: List[str]
    applied_filter_columns: List[Column]
    rejected_filter_columns: List[Column]
    status: Optional[str]
    error_message: Optional[str]
    is_loaded: bool
    stacktrace: Optional[str]
    is_cached: Optional[bool]
    cache_dttm: Optional[str]
    cache_value: Optional[Dict[str, Any]]
    sql_rowcount: Optional[int]

    def __init__(self, df=DataFrame(), query='', annotation_data=None,
        applied_template_filters=None, applied_filter_columns=None,
        rejected_filter_columns=None, status=None, error_message=None,
        is_loaded=False, stacktrace=None, is_cached=None, cache_dttm=None,
        cache_value=None, sql_rowcount=None):
        self.df = df
        self.query = query
        self.annotation_data = {
            } if annotation_data is None else annotation_data
        self.applied_template_filters = applied_template_filters or []
        self.applied_filter_columns = applied_filter_columns or []
        self.rejected_filter_columns = rejected_filter_columns or []
        self.status = status
        self.error_message = error_message
        self.is_loaded = is_loaded
        self.stacktrace = stacktrace
        self.is_cached = is_cached
        self.cache_dttm = cache_dttm
        self.cache_value = cache_value
        self.sql_rowcount = sql_rowcount

    def set_query_result(self, key, query_result, annotation_data=None,
        force_query=False, timeout=None, datasource_uid=None, region=
        CacheRegion.DEFAULT):
        """
        Set dataframe of query-result to specific cache region
        """
        try:
            self.status = query_result.status
            self.query = query_result.query
            self.applied_template_filters = (query_result.
                applied_template_filters)
            self.applied_filter_columns = query_result.applied_filter_columns
            self.rejected_filter_columns = query_result.rejected_filter_columns
            self.error_message = query_result.error_message
            self.df = query_result.df
            self.sql_rowcount = query_result.sql_rowcount
            self.annotation_data = {
                } if annotation_data is None else annotation_data
            if self.status != QueryStatus.FAILED:
                stats_logger.incr('loaded_from_source')
                if not force_query:
                    stats_logger.incr('loaded_from_source_without_force')
                self.is_loaded = True
            value: Dict[str, Any] = {'df': self.df, 'query': self.query,
                'applied_template_filters': self.applied_template_filters,
                'applied_filter_columns': self.applied_filter_columns,
                'rejected_filter_columns': self.rejected_filter_columns,
                'annotation_data': self.annotation_data, 'sql_rowcount':
                self.sql_rowcount}
            if self.is_loaded and key and self.status != QueryStatus.FAILED:
                self.set(key=key, value=value, timeout=timeout,
                    datasource_uid=datasource_uid, region=region)
        except Exception as ex:
            logger.exception(ex)
            if not self.error_message:
                self.error_message = str(ex)
            self.status = QueryStatus.FAILED
            self.stacktrace = get_stacktrace()

    @classmethod
    def get(cls, key, region=CacheRegion.DEFAULT, force_query=False,
        force_cached=False):
        """
        Initialize QueryCacheManager by query-cache key
        """
        query_cache: QueryCacheManager = cls()
        if not key or not _cache[region] or force_query:
            return query_cache
        cache_value: Optional[Dict[str, Any]] = _cache[region].get(key)
        if cache_value:
            logger.debug('Cache key: %s', key)
            stats_logger.incr('loading_from_cache')
            try:
                query_cache.df = cache_value['df']
                query_cache.query = cache_value['query']
                query_cache.annotation_data = cache_value.get('annotation_data'
                    , {})
                query_cache.applied_template_filters = cache_value.get(
                    'applied_template_filters', [])
                query_cache.applied_filter_columns = cache_value.get(
                    'applied_filter_columns', [])
                query_cache.rejected_filter_columns = cache_value.get(
                    'rejected_filter_columns', [])
                query_cache.status = QueryStatus.SUCCESS
                query_cache.is_loaded = True
                query_cache.is_cached = cache_value is not None
                query_cache.sql_rowcount = cache_value.get('sql_rowcount', None
                    )
                query_cache.cache_dttm = cache_value['dttm'
                    ] if cache_value is not None else None
                query_cache.cache_value = cache_value
                stats_logger.incr('loaded_from_cache')
            except KeyError as ex:
                logger.exception(ex)
                logger.error('Error reading cache: %s',
                    error_msg_from_exception(ex), exc_info=True)
            logger.debug('Serving from cache')
        if force_cached and not query_cache.is_loaded:
            logger.warning(
                'force_cached (QueryContext): value not found for key %s', key)
            raise CacheLoadError('Error loading data from cache')
        return query_cache

    @staticmethod
    def set(key, value, timeout=None, datasource_uid=None, region=
        CacheRegion.DEFAULT):
        """
        set value to specify cache region, proxy for `set_and_log_cache`
        """
        if key:
            set_and_log_cache(_cache[region], key, value, timeout,
                datasource_uid)

    @staticmethod
    def delete(key, region=CacheRegion.DEFAULT):
        if key:
            _cache[region].delete(key)

    @staticmethod
    def has(key, region=CacheRegion.DEFAULT):
        return bool(_cache[region].get(key)) if key else False
