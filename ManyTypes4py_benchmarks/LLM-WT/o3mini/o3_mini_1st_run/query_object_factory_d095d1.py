from __future__ import annotations
from typing import Any, Optional, Dict, List, cast, TYPE_CHECKING
from superset.common.chart_data import ChartDataResultType
from superset.common.query_object import QueryObject, QueryObjectFilterClause
from superset.common.utils.time_range_utils import get_since_until_from_time_range
from superset.constants import NO_TIME_RANGE
from superset.superset_typing import Column
from superset.utils.core import apply_max_row_limit, DatasourceDict, DatasourceType

if TYPE_CHECKING:
    from superset.connectors.sqla.models import BaseDatasource
    from superset.daos.datasource import DatasourceDAO


class QueryObjectFactory:
    def __init__(self, app_configurations: Dict[str, Any], _datasource_dao: DatasourceDAO) -> None:
        self._config: Dict[str, Any] = app_configurations
        self._datasource_dao: DatasourceDAO = _datasource_dao

    def create(
        self,
        parent_result_type: ChartDataResultType,
        datasource: Optional[DatasourceDict] = None,
        extras: Optional[Dict[str, Any]] = None,
        row_limit: Optional[int] = None,
        time_range: Optional[str] = None,
        time_shift: Optional[str] = None,
        **kwargs: Any,
    ) -> QueryObject:
        datasource_model_instance: Optional[BaseDatasource] = None
        if datasource:
            datasource_model_instance = self._convert_to_model(datasource)
        processed_extras: Dict[str, Any] = self._process_extras(extras)
        result_type: ChartDataResultType = kwargs.setdefault("result_type", parent_result_type)
        row_limit_int: int = self._process_row_limit(row_limit, result_type)
        processed_time_range: str = self._process_time_range(time_range, kwargs.get("filters"), kwargs.get("columns"))
        from_dttm, to_dttm = get_since_until_from_time_range(processed_time_range, time_shift, processed_extras)
        kwargs["from_dttm"] = from_dttm
        kwargs["to_dttm"] = to_dttm
        return QueryObject(
            datasource=datasource_model_instance,
            extras=extras,
            row_limit=row_limit_int,
            time_range=time_range,
            time_shift=time_shift,
            **kwargs,
        )

    def _convert_to_model(self, datasource: DatasourceDict) -> BaseDatasource:
        return self._datasource_dao.get_datasource(
            datasource_type=DatasourceType(datasource["type"]), datasource_id=int(datasource["id"])
        )

    def _process_extras(self, extras: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        extras = extras or {}
        return extras

    def _process_row_limit(self, row_limit: Optional[int], result_type: ChartDataResultType) -> int:
        default_row_limit: int = (
            self._config["SAMPLES_ROW_LIMIT"]
            if result_type == ChartDataResultType.SAMPLES
            else self._config["ROW_LIMIT"]
        )
        return apply_max_row_limit(row_limit or default_row_limit)

    @staticmethod
    def _process_time_range(
        time_range: Optional[str],
        filters: Optional[List[QueryObjectFilterClause]] = None,
        columns: Optional[List[Column]] = None,
    ) -> str:
        if time_range is None:
            time_range = NO_TIME_RANGE
            temporal_flt = [flt for flt in filters or [] if flt.get("op") == "TEMPORAL_RANGE"]
            if temporal_flt:
                x_axis_label = None
                if columns is not None:
                    from superset.utils.core import get_x_axis_label
                    x_axis_label = get_x_axis_label(columns)
                match_flt = [flt for flt in temporal_flt if flt.get("col") == x_axis_label]
                if match_flt:
                    time_range = cast(str, match_flt[0].get("val"))
                else:
                    time_range = cast(str, temporal_flt[0].get("val"))
        return time_range
