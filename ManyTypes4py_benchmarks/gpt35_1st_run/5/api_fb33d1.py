from __future__ import annotations
import contextlib
import logging
from typing import Any, TYPE_CHECKING, Dict, Union
from flask import current_app, g, make_response, request, Response
from flask_appbuilder.api import expose, protect
from flask_babel import gettext as _
from marshmallow import ValidationError
from superset import is_feature_enabled, security_manager
from superset.async_events.async_query_manager import AsyncQueryTokenException
from superset.charts.api import ChartRestApi
from superset.charts.client_processing import apply_client_processing
from superset.charts.data.query_context_cache_loader import QueryContextCacheLoader
from superset.charts.schemas import ChartDataQueryContextSchema
from superset.commands.chart.data.create_async_job_command import CreateAsyncChartDataJobCommand
from superset.commands.chart.data.get_data_command import ChartDataCommand
from superset.commands.chart.exceptions import ChartDataCacheLoadError, ChartDataQueryFailedError
from superset.common.chart_data import ChartDataResultFormat, ChartDataResultType
from superset.connectors.sqla.models import BaseDatasource
from superset.daos.exceptions import DatasourceNotFound
from superset.exceptions import QueryObjectValidationError
from superset.extensions import event_logger
from superset.models.sql_lab import Query
from superset.utils import json
from superset.utils.core import create_zip, DatasourceType, get_user_id
from superset.utils.decorators import logs_context
from superset.views.base import CsvResponse, generate_download_headers, XlsxResponse
from superset.views.base_api import statsd_metrics
if TYPE_CHECKING:
    from superset.common.query_context import QueryContext

class ChartDataRestApi(ChartRestApi):
    include_route_methods: set[str] = {'get_data', 'data', 'data_from_cache'}

    def get_data(self, pk: int) -> Response:
        ...

    def data(self) -> Response:
        ...

    def data_from_cache(self, cache_key: str) -> Response:
        ...

    def _run_async(self, form_data: Dict[str, Any], command: ChartDataCommand) -> Response:
        ...

    def _send_chart_response(self, result: Dict[str, Any], form_data: Dict[str, Any] = None, datasource: BaseDatasource = None) -> Response:
        ...

    def _get_data_response(self, command: ChartDataCommand, force_cached: bool = False, form_data: Dict[str, Any] = None, datasource: BaseDatasource = None) -> Response:
        ...

    def _load_query_context_form_from_cache(self, cache_key: str) -> Dict[str, Any]:
        ...

    def _map_form_data_datasource_to_dataset_id(self, form_data: Dict[str, Any]) -> Dict[str, Union[int, str, None]]:
        ...

    def _create_query_context_from_form(self, form_data: Dict[str, Any]) -> QueryContext:
        ...
