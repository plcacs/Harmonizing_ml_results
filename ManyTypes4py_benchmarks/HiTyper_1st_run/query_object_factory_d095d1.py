from __future__ import annotations
from typing import Any, cast, TYPE_CHECKING
from superset.common.chart_data import ChartDataResultType
from superset.common.query_object import QueryObject
from superset.common.utils.time_range_utils import get_since_until_from_time_range
from superset.constants import NO_TIME_RANGE
from superset.superset_typing import Column
from superset.utils.core import apply_max_row_limit, DatasourceDict, DatasourceType, FilterOperator, get_x_axis_label, QueryObjectFilterClause
if TYPE_CHECKING:
    from superset.connectors.sqla.models import BaseDatasource
    from superset.daos.datasource import DatasourceDAO

class QueryObjectFactory:

    def __init__(self, app_configurations: Any, _datasource_dao: Union[ContextManager, dict, dict[str, models.DeviceRow]]) -> None:
        self._config = app_configurations
        self._datasource_dao = _datasource_dao

    def create(self, parent_result_type: Union[str, dict[str, typing.Any], dict], datasource: Union[None, str, tartare.core.models.ValidityPeriod, dict]=None, extras: Union[None, str, int]=None, row_limit: Union[None, int, float, dict[int, dict]]=None, time_range: Union[None, str, float]=None, time_shift: Union[None, str]=None, **kwargs) -> QueryObject:
        datasource_model_instance = None
        if datasource:
            datasource_model_instance = self._convert_to_model(datasource)
        processed_extras = self._process_extras(extras)
        result_type = kwargs.setdefault('result_type', parent_result_type)
        row_limit = self._process_row_limit(row_limit, result_type)
        processed_time_range = self._process_time_range(time_range, kwargs.get('filters'), kwargs.get('columns'))
        from_dttm, to_dttm = get_since_until_from_time_range(processed_time_range, time_shift, processed_extras)
        kwargs['from_dttm'] = from_dttm
        kwargs['to_dttm'] = to_dttm
        return QueryObject(datasource=datasource_model_instance, extras=extras, row_limit=row_limit, time_range=time_range, time_shift=time_shift, **kwargs)

    def _convert_to_model(self, datasource: Union[str, dict, annofabapi.models.Inspection]) -> Union[str, bool]:
        return self._datasource_dao.get_datasource(datasource_type=DatasourceType(datasource['type']), datasource_id=int(datasource['id']))

    def _process_extras(self, extras: dict[str, typing.Any]) -> Union[dict[str, typing.Any], dict]:
        extras = extras or {}
        return extras

    def _process_row_limit(self, row_limit: Union[int, str], result_type: Union[str, dict, int]) -> Union[bool, fonduer.parser.models.table.Cell, fonduer.parser.models.sentence.Sentence]:
        default_row_limit = self._config['SAMPLES_ROW_LIMIT'] if result_type == ChartDataResultType.SAMPLES else self._config['ROW_LIMIT']
        return apply_max_row_limit(row_limit or default_row_limit)

    @staticmethod
    def _process_time_range(time_range: Union[str, dict, dict[str, typing.Any]], filters: Union[None, int, typing.Sequence[str], typing.Container]=None, columns: Union[None, list[str], str]=None) -> Union[str, list[str], list]:
        if time_range is None:
            time_range = NO_TIME_RANGE
            temporal_flt = [flt for flt in filters or [] if flt.get('op') == FilterOperator.TEMPORAL_RANGE]
            if temporal_flt:
                x_axis_label = get_x_axis_label(columns)
                match_flt = [flt for flt in temporal_flt if flt.get('col') == x_axis_label]
                if match_flt:
                    time_range = cast(str, match_flt[0].get('val'))
                else:
                    time_range = cast(str, temporal_flt[0].get('val'))
        return time_range