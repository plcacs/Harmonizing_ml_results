from __future__ import annotations
import logging
from typing import Any, ClassVar, TYPE_CHECKING
import pandas as pd
from superset.common.chart_data import ChartDataResultFormat, ChartDataResultType
from superset.common.query_context_processor import CachedTimeOffset, QueryContextProcessor
from superset.common.query_object import QueryObject
from superset.models.slice import Slice
from superset.utils.core import GenericDataType
if TYPE_CHECKING:
    from superset.connectors.sqla.models import BaseDatasource
    from superset.models.helpers import QueryResult
logger = logging.getLogger(__name__)


class QueryContext:
    """
    The query context contains the query object and additional fields necessary
    to retrieve the data payload for a given viz.
    """
    cache_type: ClassVar[str] = 'df'
    enforce_numerical_metrics: ClassVar[bool] = True
    datasource: BaseDatasource
    slice_: Slice | None = None
    queries: list[QueryObject]
    form_data: dict[str, Any] | None
    result_type: ChartDataResultType
    result_format: ChartDataResultFormat
    force: bool
    custom_cache_timeout: int | None
    cache_values: dict[str, Any]
    _processor: QueryContextProcessor

    def __init__(self, *, datasource: BaseDatasource, queries: list[
        QueryObject], slice_: (Slice | None), form_data: (dict[str, Any] |
        None), result_type: ChartDataResultType, result_format:
        ChartDataResultFormat, force: bool=False, custom_cache_timeout: (
        int | None)=None, cache_values: dict[str, Any]):
        self.datasource = datasource
        self.slice_ = slice_
        self.result_type = result_type
        self.result_format = result_format
        self.queries = queries
        self.form_data = form_data
        self.force = force
        self.custom_cache_timeout = custom_cache_timeout
        self.cache_values = cache_values
        self._processor = QueryContextProcessor(self)

    def get_data(self, df, coltypes):
        return self._processor.get_data(df, coltypes)

    def get_payload(self, cache_query_context=False, force_cached=False):
        """Returns the query results with both metadata and data"""
        return self._processor.get_payload(cache_query_context, force_cached)

    def get_cache_timeout(self):
        if self.custom_cache_timeout is not None:
            return self.custom_cache_timeout
        if self.slice_ and self.slice_.cache_timeout is not None:
            return self.slice_.cache_timeout
        if self.datasource.cache_timeout is not None:
            return self.datasource.cache_timeout
        if hasattr(self.datasource, 'database'):
            return self.datasource.database.cache_timeout
        return None

    def query_cache_key(self, query_obj, **kwargs: Any):
        return self._processor.query_cache_key(query_obj, **kwargs)

    def get_df_payload(self, query_obj, force_cached=False):
        return self._processor.get_df_payload(query_obj=query_obj,
            force_cached=force_cached)

    def get_query_result(self, query_object):
        return self._processor.get_query_result(query_object)

    def processing_time_offsets(self, df, query_object):
        return self._processor.processing_time_offsets(df, query_object)

    def raise_for_access(self):
        self._processor.raise_for_access()
