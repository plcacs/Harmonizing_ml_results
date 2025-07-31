from __future__ import annotations
import logging
from typing import Any, ClassVar, Dict, List, Optional, TYPE_CHECKING
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
    slice_: Optional[Slice] = None

    def __init__(
        self,
        *,
        datasource: BaseDatasource,
        queries: List[QueryObject],
        slice_: Optional[Slice],
        form_data: Dict[str, Any],
        result_type: ChartDataResultType,
        result_format: ChartDataResultFormat,
        force: bool = False,
        custom_cache_timeout: Optional[int] = None,
        cache_values: GenericDataType,
    ) -> None:
        self.datasource: BaseDatasource = datasource
        self.slice_: Optional[Slice] = slice_
        self.result_type: ChartDataResultType = result_type
        self.result_format: ChartDataResultFormat = result_format
        self.queries: List[QueryObject] = queries
        self.form_data: Dict[str, Any] = form_data
        self.force: bool = force
        self.custom_cache_timeout: Optional[int] = custom_cache_timeout
        self.cache_values: GenericDataType = cache_values
        self._processor: QueryContextProcessor = QueryContextProcessor(self)

    def get_data(self, df: pd.DataFrame, coltypes: Any) -> Any:
        return self._processor.get_data(df, coltypes)

    def get_payload(self, cache_query_context: bool = False, force_cached: bool = False) -> Any:
        """Returns the query results with both metadata and data"""
        return self._processor.get_payload(cache_query_context, force_cached)

    def get_cache_timeout(self) -> Optional[int]:
        if self.custom_cache_timeout is not None:
            return self.custom_cache_timeout
        if self.slice_ and self.slice_.cache_timeout is not None:
            return self.slice_.cache_timeout
        if self.datasource.cache_timeout is not None:
            return self.datasource.cache_timeout
        if hasattr(self.datasource, 'database'):
            return self.datasource.database.cache_timeout
        return None

    def query_cache_key(self, query_obj: QueryObject, **kwargs: Any) -> Any:
        return self._processor.query_cache_key(query_obj, **kwargs)

    def get_df_payload(self, query_obj: QueryObject, force_cached: bool = False) -> Any:
        return self._processor.get_df_payload(query_obj=query_obj, force_cached=force_cached)

    def get_query_result(self, query_object: QueryObject) -> QueryResult:
        return self._processor.get_query_result(query_object)

    def processing_time_offsets(self, df: pd.DataFrame, query_object: QueryObject) -> CachedTimeOffset:
        return self._processor.processing_time_offsets(df, query_object)

    def raise_for_access(self) -> None:
        self._processor.raise_for_access()