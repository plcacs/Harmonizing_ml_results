from __future__ import annotations
import functools
import logging
from typing import Any, Callable, cast, Optional, Dict, List, Tuple, TypeVar
from flask import request, Response
from flask_appbuilder import Model, ModelRestApi
from flask_appbuilder.api import BaseApi, expose, protect, rison, safe
from flask_appbuilder.models.filters import BaseFilter, Filters
from flask_appbuilder.models.sqla.filters import FilterStartsWith
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_babel import lazy_gettext as _
from marshmallow import fields, Schema
from sqlalchemy import and_, distinct, func
from sqlalchemy.orm.query import Query
from superset import is_feature_enabled
from superset.exceptions import InvalidPayloadFormatError
from superset.extensions import db, event_logger, security_manager, stats_logger_manager
from superset.models.core import FavStar
from superset.models.dashboard import Dashboard
from superset.models.slice import Slice
from superset.schemas import error_payload_content
from superset.sql_lab import Query as SqllabQuery
from superset.superset_typing import FlaskResponse
from superset.utils.core import get_user_id, time_function
from superset.views.error_handling import handle_api_exception

logger = logging.getLogger(__name__)
get_related_schema: Dict[str, Any] = {
    'type': 'object',
    'properties': {
        'page_size': {'type': 'integer'},
        'page': {'type': 'integer'},
        'include_ids': {'type': 'array', 'items': {'type': 'integer'}},
        'filter': {'type': 'string'}
    }
}

class RelatedResultResponseSchema(Schema):
    value = fields.Integer(metadata={'description': 'The related item identifier'})
    text = fields.String(metadata={'description': 'The related item string representation'})
    extra = fields.Dict(metadata={'description': 'The extra metadata for related item'})

class RelatedResponseSchema(Schema):
    count = fields.Integer(metadata={'description': 'The total number of related values'})
    result = fields.List(fields.Nested(RelatedResultResponseSchema))

class DistinctResultResponseSchema(Schema):
    text = fields.String(metadata={'description': 'The distinct item'})

class DistincResponseSchema(Schema):
    count = fields.Integer(metadata={'description': 'The total number of distinct values'})
    result = fields.List(fields.Nested(DistinctResultResponseSchema))

F = TypeVar("F", bound=Callable[..., Any])

def requires_json(f: F) -> F:
    """
    Require JSON-like formatted request to the REST API
    """
    def wraps(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not request.is_json:
            raise InvalidPayloadFormatError(message='Request is not JSON')
        return f(self, *args, **kwargs)
    return functools.update_wrapper(wraps, f)  # type: ignore

def requires_form_data(f: F) -> F:
    """
    Require 'multipart/form-data' as request MIME type
    """
    def wraps(self: Any, *args: Any, **kwargs: Any) -> Any:
        if request.mimetype != 'multipart/form-data':
            raise InvalidPayloadFormatError(message="Request MIME type is not 'multipart/form-data'")
        return f(self, *args, **kwargs)
    return functools.update_wrapper(wraps, f)  # type: ignore

def statsd_metrics(f: F) -> F:
    """
    Handle sending all statsd metrics from the REST API
    """
    def wraps(self: BaseSupersetApiMixin, *args: Any, **kwargs: Any) -> Any:
        func_name: str = f.__name__
        try:
            duration, response = time_function(f, self, *args, **kwargs)
        except Exception as ex:
            if hasattr(ex, 'status') and ex.status < 500:
                self.incr_stats('warning', func_name)
            else:
                self.incr_stats('error', func_name)
            raise
        self.send_stats_metrics(response, func_name, duration)
        return response
    return functools.update_wrapper(wraps, f)  # type: ignore

def validate_feature_flags(feature_flags: List[str]) -> Callable[[F], F]:
    """
    A decorator to check if all given feature flags are enabled.

    :param feature_flags: List of feature flag names to be checked.
    """
    def decorate(f: F) -> F:
        @functools.wraps(f)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            if not all(is_feature_enabled(flag) for flag in feature_flags):
                return self.response_404()
            return f(self, *args, **kwargs)
        return wrapper  # type: ignore
    return decorate

class RelatedFieldFilter:
    def __init__(self, field_name: str, filter_class: Any) -> None:
        self.field_name: str = field_name
        self.filter_class: Any = filter_class

class BaseFavoriteFilter(BaseFilter):
    """
    Base Custom filter for the GET list that filters all dashboards, slices
    that a user has favored or not
    """
    name = _('Is favorite')
    arg_name = ''
    class_name = ''
    ' The FavStar class_name to user '
    model = Dashboard
    ' The SQLAlchemy model '

    def apply(self, query: Query, value: Any) -> Query:
        if security_manager.current_user is None:
            return query
        users_favorite_query = db.session.query(FavStar.obj_id).filter(
            and_(FavStar.user_id == get_user_id(), FavStar.class_name == self.class_name)
        )
        if value:
            return query.filter(and_(self.model.id.in_(users_favorite_query)))
        return query.filter(and_(~self.model.id.in_(users_favorite_query)))

class BaseSupersetApiMixin:
    csrf_exempt: bool = False
    responses: Dict[str, Any] = {
        '400': {'description': 'Bad request', 'content': error_payload_content},
        '401': {'description': 'Unauthorized', 'content': error_payload_content},
        '403': {'description': 'Forbidden', 'content': error_payload_content},
        '404': {'description': 'Not found', 'content': error_payload_content},
        '410': {'description': 'Gone', 'content': error_payload_content},
        '422': {'description': 'Could not process entity', 'content': error_payload_content},
        '500': {'description': 'Fatal error', 'content': error_payload_content},
    }

    def incr_stats(self, action: str, func_name: str) -> None:
        """
        Proxy function for statsd.incr to impose a key structure for REST API's
        :param action: String with an action name eg: error, success
        :param func_name: The function name
        """
        stats_logger_manager.instance.incr(f'{self.__class__.__name__}.{func_name}.{action}')

    def timing_stats(self, action: str, func_name: str, value: float) -> None:
        """
        Proxy function for statsd.incr to impose a key structure for REST API's
        :param action: String with an action name eg: error, success
        :param func_name: The function name
        :param value: A float with the time it took for the endpoint to execute
        """
        stats_logger_manager.instance.timing(f'{self.__class__.__name__}.{func_name}.{action}', value)

    def send_stats_metrics(self, response: Response, key: str, time_delta: Optional[float] = None) -> None:
        """
        Helper function to handle sending statsd metrics
        :param response: flask response object, will evaluate if it was an error
        :param key: The function name
        :param time_delta: Optional time it took for the endpoint to execute
        """
        if 200 <= response.status_code < 400:
            self.incr_stats('success', key)
        elif 400 <= response.status_code < 500:
            self.incr_stats('warning', key)
        else:
            self.incr_stats('error', key)
        if time_delta is not None:
            self.timing_stats('time', key, time_delta)

class BaseSupersetApi(BaseSupersetApiMixin, BaseApi):
    pass

class BaseSupersetModelRestApi(BaseSupersetApiMixin, ModelRestApi):
    """
    Extends FAB's ModelResApi to implement specific superset generic functionality
    """
    method_permission_name: Dict[str, str] = {
        'bulk_delete': 'delete', 'data': 'list', 'data_from_cache': 'list', 'delete': 'delete',
        'distinct': 'list', 'export': 'mulexport', 'import_': 'add', 'get': 'show', 'get_list': 'list',
        'info': 'list', 'post': 'add', 'put': 'edit', 'refresh': 'edit', 'related': 'list',
        'related_objects': 'list', 'schemas': 'list', 'select_star': 'list', 'table_metadata': 'list',
        'test_connection': 'post', 'thumbnail': 'list', 'viz_types': 'list'
    }
    order_rel_fields: Dict[str, Tuple[str, str]] = {}
    '\n    Impose ordering on related fields query::\n\n        order_rel_fields = {\n            "<RELATED_FIELD>": ("<RELATED_FIELD_FIELD>", "<asc|desc>"),\n             ...\n        }\n    '
    base_related_field_filters: Dict[str, Any] = {}
    '\n    This is used to specify a base filter for related fields\n    when they are accessed through the \'/related/<column_name>\' endpoint.\n    When combined with the `related_field_filters` attribute,\n    this filter will be applied in addition to the latest::\n\n        base_related_field_filters = {\n            "<RELATED_FIELD>": "<FILTER>")\n        }\n    '
    related_field_filters: Dict[str, Any] = {}
    '\n    Specify a filter for related fields when they are accessed\n    through the \'/related/<column_name>\' endpoint.\n    When combined with the `base_related_field_filters` attribute,\n    this filter will be applied in prior to the latest::\n\n        related_fields = {\n            "<RELATED_FIELD>": <RelatedFieldFilter>)\n        }\n    '
    allowed_rel_fields: set = set()
    text_field_rel_fields: Dict[str, str] = {}
    '\n    Declare an alternative for the human readable representation of the Model object::\n\n        text_field_rel_fields = {\n            "<RELATED_FIELD>": "<RELATED_OBJECT_FIELD>"\n        }\n    '
    extra_fields_rel_fields: Dict[str, List[str]] = {'owners': ['email', 'active']}
    '\n    Declare extra fields for the representation of the Model object::\n\n        extra_fields_rel_fields = {\n            "<RELATED_FIELD>": "[<RELATED_OBJECT_FIELD_1>, <RELATED_OBJECT_FIELD_2>]"\n        }\n    '
    allowed_distinct_fields: set = set()

    def __init__(self) -> None:
        super().__init__()
        if self.apispec_parameter_schemas is None:
            self.apispec_parameter_schemas = {}
        self.apispec_parameter_schemas['get_related_schema'] = get_related_schema
        self.openapi_spec_component_schemas = self.openapi_spec_component_schemas + (RelatedResponseSchema, DistincResponseSchema)

    def _init_properties(self) -> None:
        """
        Lock down initial not configured REST API columns. We want to just expose
        model ids, if something is misconfigured. By default FAB exposes all available
        columns on a Model
        """
        model_id: str = self.datamodel.get_pk_name()
        if self.list_columns is None and (not self.list_model_schema):
            self.list_columns = [model_id]
        if self.show_columns is None and (not self.show_model_schema):
            self.show_columns = [model_id]
        if self.edit_columns is None and (not self.edit_model_schema):
            self.edit_columns = [model_id]
        if self.add_columns is None and (not self.add_model_schema):
            self.add_columns = [model_id]
        super()._init_properties()

    def _get_related_filter(self, datamodel: Any, column_name: str, value: Any) -> Filters:
        filter_field: Optional[RelatedFieldFilter] = self.related_field_filters.get(column_name)
        if isinstance(filter_field, str):
            filter_field = RelatedFieldFilter(cast(str, filter_field), FilterStartsWith)
        filter_field = cast(Optional[RelatedFieldFilter], filter_field)
        search_columns: Optional[List[str]] = [filter_field.field_name] if filter_field else None
        filters: Filters = datamodel.get_filters(search_columns)
        base_filters = self.base_related_field_filters.get(column_name)
        if base_filters:
            filters.add_filter_list(base_filters)
        if value and filter_field:
            filters.add_filter(filter_field.field_name, filter_field.filter_class, value)
        return filters

    def _get_distinct_filter(self, column_name: str, value: Any) -> Filters:
        filter_field: RelatedFieldFilter = RelatedFieldFilter(column_name, FilterStartsWith)
        filter_field = cast(RelatedFieldFilter, filter_field)
        search_columns: Optional[List[str]] = [filter_field.field_name] if filter_field else None
        filters: Filters = self.datamodel.get_filters(search_columns)
        filters.add_filter_list(self.base_filters)
        if value and filter_field:
            filters.add_filter(filter_field.field_name, filter_field.filter_class, value)
        return filters

    def _get_text_for_model(self, model: Any, column_name: str) -> str:
        if column_name in self.text_field_rel_fields:
            model_column_name: Optional[str] = self.text_field_rel_fields.get(column_name)
            if model_column_name:
                return getattr(model, model_column_name)
        return str(model)

    def _get_extra_field_for_model(self, model: Any, column_name: str) -> Dict[str, Any]:
        ret: Dict[str, Any] = {}
        if column_name in self.extra_fields_rel_fields:
            model_column_names: Optional[List[str]] = self.extra_fields_rel_fields.get(column_name)
            if model_column_names:
                for key in model_column_names:
                    ret[key] = getattr(model, key)
        return ret

    def _get_result_from_rows(self, datamodel: Any, rows: List[Any], column_name: str) -> List[Dict[str, Any]]:
        return [
            {
                'value': datamodel.get_pk_value(row),
                'text': self._get_text_for_model(row, column_name),
                'extra': self._get_extra_field_for_model(row, column_name),
            }
            for row in rows
        ]

    def _add_extra_ids_to_result(self, datamodel: Any, column_name: str, ids: List[int], result: List[Dict[str, Any]]) -> None:
        if ids:
            values: List[Any] = [row['value'] for row in result]
            ids = [id_ for id_ in ids if id_ not in values]
            pk_col = datamodel.get_pk()
            extra_rows = db.session.query(datamodel.obj).filter(pk_col.in_(ids)).all()
            result += self._get_result_from_rows(datamodel, extra_rows, column_name)

    @event_logger.log_this_with_context(action=lambda self, *args, **kwargs: f'{self.__class__.__name__}.info', object_ref=False, log_to_statsd=False)
    @handle_api_exception
    def info_headless(self, **kwargs: Any) -> FlaskResponse:
        """
        Add statsd metrics to builtin FAB _info endpoint
        """
        duration, response = time_function(super().info_headless, **kwargs)
        self.send_stats_metrics(response, self.info.__name__, duration)
        return response

    @event_logger.log_this_with_context(action=lambda self, *args, **kwargs: f'{self.__class__.__name__}.get', object_ref=False, log_to_statsd=False)
    @handle_api_exception
    def get_headless(self, pk: Any, **kwargs: Any) -> FlaskResponse:
        """
        Add statsd metrics to builtin FAB GET endpoint
        """
        duration, response = time_function(super().get_headless, pk, **kwargs)
        self.send_stats_metrics(response, self.get.__name__, duration)
        return response

    @event_logger.log_this_with_context(action=lambda self, *args, **kwargs: f'{self.__class__.__name__}.get_list', object_ref=False, log_to_statsd=False)
    @handle_api_exception
    def get_list_headless(self, **kwargs: Any) -> FlaskResponse:
        """
        Add statsd metrics to builtin FAB GET list endpoint
        """
        duration, response = time_function(super().get_list_headless, **kwargs)
        self.send_stats_metrics(response, self.get_list.__name__, duration)
        return response

    @event_logger.log_this_with_context(action=lambda self, *args, **kwargs: f'{self.__class__.__name__}.post', object_ref=False, log_to_statsd=False)
    @handle_api_exception
    def post_headless(self) -> FlaskResponse:
        """
        Add statsd metrics to builtin FAB POST endpoint
        """
        duration, response = time_function(super().post_headless)
        self.send_stats_metrics(response, self.post.__name__, duration)
        return response

    @event_logger.log_this_with_context(action=lambda self, *args, **kwargs: f'{self.__class__.__name__}.put', object_ref=False, log_to_statsd=False)
    @handle_api_exception
    def put_headless(self, pk: Any) -> FlaskResponse:
        """
        Add statsd metrics to builtin FAB PUT endpoint
        """
        duration, response = time_function(super().put_headless, pk)
        self.send_stats_metrics(response, self.put.__name__, duration)
        return response

    @event_logger.log_this_with_context(action=lambda self, *args, **kwargs: f'{self.__class__.__name__}.delete', object_ref=False, log_to_statsd=False)
    @handle_api_exception
    def delete_headless(self, pk: Any) -> FlaskResponse:
        """
        Add statsd metrics to builtin FAB DELETE endpoint
        """
        duration, response = time_function(super().delete_headless, pk)
        self.send_stats_metrics(response, self.delete.__name__, duration)
        return response

    @expose('/related/<column_name>', methods=('GET',))
    @protect()
    @safe
    @statsd_metrics
    @rison(get_related_schema)
    @handle_api_exception
    def related(self, column_name: str, **kwargs: Any) -> FlaskResponse:
        """Get related fields data.
        ---
        get:
          summary: Get related fields data
          parameters:
          - in: path
            schema:
              type: string
            name: column_name
          - in: query
            name: q
            content:
              application/json:
                schema:
                  $ref: '#/components/schemas/get_related_schema'
          responses:
            200:
              description: Related column data
              content:
                application/json:
                  schema:
                  schema:
                    $ref: "#/components/schemas/RelatedResponseSchema"
            400:
              $ref: '#/components/responses/400'
            401:
              $ref: '#/components/responses/401'
            404:
              $ref: '#/components/responses/404'
            500:
              $ref: '#/components/responses/500'
        """
        if column_name not in self.allowed_rel_fields:
            self.incr_stats('error', self.related.__name__)
            return self.response_404()
        args: Dict[str, Any] = kwargs.get('rison', {})
        page, page_size = self._handle_page_args(args)
        ids: Optional[List[int]] = args.get('include_ids')
        if page and ids:
            return self.response_422()
        try:
            datamodel = self.datamodel.get_related_interface(column_name)
        except KeyError:
            return self.response_404()
        page, page_size = self._sanitize_page_args(page, page_size)
        if (order_field := self.order_rel_fields.get(column_name)):
            order_column, order_direction = order_field
        else:
            order_column, order_direction = ('', '')
        filters: Filters = self._get_related_filter(datamodel, column_name, args.get('filter'))
        total_rows, rows = datamodel.query(filters, order_column, order_direction, page=page, page_size=page_size)
        result: List[Dict[str, Any]] = self._get_result_from_rows(datamodel, rows, column_name)
        if ids:
            self._add_extra_ids_to_result(datamodel, column_name, ids, result)
            total_rows = len(result)
        return self.response(200, count=total_rows, result=result)

    @expose('/distinct/<column_name>', methods=('GET',))
    @protect()
    @safe
    @statsd_metrics
    @rison(get_related_schema)
    @handle_api_exception
    def distinct(self, column_name: str, **kwargs: Any) -> FlaskResponse:
        """Get distinct values from field data.
        ---
        get:
          summary: Get distinct values from field data
          parameters:
          - in: path
            schema:
              type: string
            name: column_name
          - in: query
            name: q
            content:
              application/json:
                schema:
                  $ref: '#/components/schemas/get_related_schema'
          responses:
            200:
              description: Distinct field data
              content:
                application/json:
                  schema:
                  schema:
                    $ref: "#/components/schemas/DistincResponseSchema"
            400:
              $ref: '#/components/responses/400'
            401:
              $ref: '#/components/responses/401'
            404:
              $ref: '#/components/responses/404'
            500:
              $ref: '#/components/responses/500'
        """
        if column_name not in self.allowed_distinct_fields:
            self.incr_stats('error', self.related.__name__)
            return self.response_404()
        args: Dict[str, Any] = kwargs.get('rison', {})
        page, page_size = self._sanitize_page_args(*self._handle_page_args(args))
        filters: Filters = self._get_distinct_filter(column_name, args.get('filter'))
        query_count = self.appbuilder.get_session.query(func.count(distinct(getattr(self.datamodel.obj, column_name))))
        count: int = self.datamodel.apply_filters(query_count, filters).scalar()
        if count == 0:
            return self.response(200, count=count, result=[])
        query = self.appbuilder.get_session.query(distinct(getattr(self.datamodel.obj, column_name)))
        query = self.datamodel.apply_filters(query, filters)
        query = self.datamodel.apply_order_by(query, column_name, 'asc')
        result_raw = self.datamodel.apply_pagination(query, page, page_size).all()
        result: List[Dict[str, Any]] = [{'text': item[0], 'value': item[0]} for item in result_raw if item[0] is not None]
        return self.response(200, count=count, result=result)