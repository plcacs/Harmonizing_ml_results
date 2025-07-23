from __future__ import annotations
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union
from superset import app
from superset.common.chart_data import ChartDataResultFormat, ChartDataResultType
from superset.common.query_context import QueryContext
from superset.common.query_object import QueryObject
from superset.common.query_object_factory import QueryObjectFactory
from superset.daos.chart import ChartDAO
from superset.daos.datasource import DatasourceDAO
from superset.models.slice import Slice
from superset.utils.core import DatasourceDict, DatasourceType, is_adhoc_column
if TYPE_CHECKING:
    from superset.connectors.sqla.models import BaseDatasource
config = app.config

def create_query_object_factory() -> QueryObjectFactory:
    return QueryObjectFactory(config, DatasourceDAO())

class QueryContextFactory:

    def __init__(self) -> None:
        self._query_object_factory = create_query_object_factory()

    def create(
        self, 
        *, 
        datasource: DatasourceDict, 
        queries: List[Dict[str, Any]], 
        form_data: Optional[Dict[str, Any]] = None, 
        result_type: Optional[ChartDataResultType] = None, 
        result_format: Optional[ChartDataResultFormat] = None, 
        force: bool = False, 
        custom_cache_timeout: Optional[int] = None
    ) -> QueryContext:
        datasource_model_instance: Optional[BaseDatasource] = None
        if datasource:
            datasource_model_instance = self._convert_to_model(datasource)
        slice_: Optional[Slice] = None
        if form_data and form_data.get('slice_id') is not None:
            slice_ = self._get_slice(form_data.get('slice_id'))
        result_type = result_type or ChartDataResultType.FULL
        result_format = result_format or ChartDataResultFormat.JSON
        queries_: List[QueryObject] = [self._process_query_object(datasource_model_instance, form_data, self._query_object_factory.create(result_type, datasource=datasource, **query_obj)) for query_obj in queries]
        cache_values: Dict[str, Any] = {'datasource': datasource, 'queries': queries, 'result_type': result_type, 'result_format': result_format}
        return QueryContext(datasource=datasource_model_instance, queries=queries_, slice_=slice_, form_data=form_data, result_type=result_type, result_format=result_format, force=force, custom_cache_timeout=custom_cache_timeout, cache_values=cache_values)

    def _convert_to_model(self, datasource: DatasourceDict) -> BaseDatasource:
        return DatasourceDAO.get_datasource(datasource_type=DatasourceType(datasource['type']), datasource_id=int(datasource['id']))

    def _get_slice(self, slice_id: int) -> Slice:
        return ChartDAO.find_by_id(slice_id)

    def _process_query_object(self, datasource: Optional[BaseDatasource], form_data: Optional[Dict[str, Any]], query_object: QueryObject) -> QueryObject:
        self._apply_granularity(query_object, form_data, datasource)
        self._apply_filters(query_object)
        return query_object

    def _apply_granularity(self, query_object: QueryObject, form_data: Optional[Dict[str, Any]], datasource: Optional[BaseDatasource]) -> None:
        if not datasource:
            return
        temporal_columns: set = {column['column_name'] if isinstance(column, dict) else column.column_name for column in datasource.columns if (column['is_dttm'] if isinstance(column, dict) else column.is_dttm)}
        x_axis: Optional[Union[str, Dict[str, Any]]] = form_data and form_data.get('x_axis')
        if (granularity := query_object.granularity):
            filter_to_remove: Optional[Union[str, Dict[str, Any]]] = None
            if is_adhoc_column(x_axis):
                x_axis = x_axis.get('sqlExpression')  # type: ignore
            if x_axis and x_axis in temporal_columns:
                filter_to_remove = x_axis
                x_axis_column: Optional[Union[str, Dict[str, Any]]] = next((column for column in query_object.columns if column == x_axis or (isinstance(column, dict) and column['sqlExpression'] == x_axis)), None)
                if x_axis_column:
                    if isinstance(x_axis_column, dict):
                        x_axis_column['sqlExpression'] = granularity
                        x_axis_column['label'] = granularity
                    else:
                        query_object.columns = [granularity if column == x_axis_column else column for column in query_object.columns]
                    for post_processing in query_object.post_processing:
                        if post_processing.get('operation') == 'pivot':
                            post_processing['options']['index'] = [granularity]
            if not filter_to_remove:
                temporal_filters: List[Union[str, Dict[str, Any]]] = [filter['col'] for filter in query_object.filter if filter['op'] == 'TEMPORAL_RANGE']
                if len(temporal_filters) > 0:
                    if granularity in temporal_filters:
                        filter_to_remove = granularity
                    else:
                        filter_to_remove = temporal_filters[0]
            if is_adhoc_column(filter_to_remove):
                filter_to_remove = filter_to_remove.get('sqlExpression')  # type: ignore
            if filter_to_remove:
                query_object.filter = [filter for filter in query_object.filter if filter['col'] != filter_to_remove]

    def _apply_filters(self, query_object: QueryObject) -> None:
        if query_object.time_range:
            for filter_object in query_object.filter:
                if filter_object['op'] == 'TEMPORAL_RANGE':
                    filter_object['val'] = query_object.time_range
