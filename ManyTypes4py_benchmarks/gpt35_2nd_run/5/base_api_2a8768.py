from __future__ import annotations
import functools
import logging
from typing import Any, Callable, cast, List, Dict
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

get_related_schema: Dict[str, Any] = {'type': 'object', 'properties': {'page_size': {'type': 'integer'}, 'page': {'type': 'integer'}, 'include_ids': {'type': 'array', 'items': {'type': 'integer'}}, 'filter': {'type': 'string'}}}

class RelatedResultResponseSchema(Schema):
    value: fields.Integer(metadata={'description': 'The related item identifier'})
    text: fields.String(metadata={'description': 'The related item string representation'})
    extra: fields.Dict(metadata={'description': 'The extra metadata for related item'})

class RelatedResponseSchema(Schema):
    count: fields.Integer(metadata={'description': 'The total number of related values'})
    result: fields.List(fields.Nested(RelatedResultResponseSchema))

class DistinctResultResponseSchema(Schema):
    text: fields.String(metadata={'description': 'The distinct item'})

class DistincResponseSchema(Schema):
    count: fields.Integer(metadata={'description': 'The total number of distinct values'})
    result: fields.List(fields.Nested(DistinctResultResponseSchema))

def requires_json(f: Callable) -> Callable:
    def wraps(self, *args, **kwargs):
        if not request.is_json:
            raise InvalidPayloadFormatError(message='Request is not JSON')
        return f(self, *args, **kwargs)
    return functools.update_wrapper(wraps, f)

def requires_form_data(f: Callable) -> Callable:
    def wraps(self, *args, **kwargs):
        if not request.mimetype == 'multipart/form-data':
            raise InvalidPayloadFormatError(message="Request MIME type is not 'multipart/form-data'")
        return f(self, *args, **kwargs)
    return functools.update_wrapper(wraps, f)

def statsd_metrics(f: Callable) -> Callable:
    def wraps(self, *args, **kwargs):
        func_name = f.__name__
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
    return functools.update_wrapper(wraps, f)

def validate_feature_flags(feature_flags: List[str]) -> Callable:
    def decorate(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(self, *args, **kwargs):
            if not all((is_feature_enabled(flag) for flag in feature_flags)):
                return self.response_404()
            return f(self, *args, **kwargs)
        return wrapper
    return decorate

class RelatedFieldFilter:

    def __init__(self, field_name: str, filter_class: Any):
        self.field_name = field_name
        self.filter_class = filter_class

class BaseFavoriteFilter(BaseFilter):
    name = _('Is favorite')
    arg_name = ''
    class_name = ''
    model = Dashboard

    def apply(self, query, value):
        if security_manager.current_user is None:
            return query
        users_favorite_query = db.session.query(FavStar.obj_id).filter(and_(FavStar.user_id == get_user_id(), FavStar.class_name == self.class_name))
        if value:
            return query.filter(and_(self.model.id.in_(users_favorite_query)))
        return query.filter(and_(~self.model.id.in_(users_favorite_query)))

class BaseSupersetApiMixin:
    csrf_exempt: bool = False
    responses: Dict[str, Dict[str, Any]] = {'400': {'description': 'Bad request', 'content': error_payload_content}, '401': {'description': 'Unauthorized', 'content': error_payload_content}, '403': {'description': 'Forbidden', 'content': error_payload_content}, '404': {'description': 'Not found', 'content': error_payload_content}, '410': {'description': 'Gone', 'content': error_payload_content}, '422': {'description': 'Could not process entity', 'content': error_payload_content}, '500': {'description': 'Fatal error', 'content': error_payload_content}}

    def incr_stats(self, action: str, func_name: str) -> None:
        ...

    def timing_stats(self, action: str, func_name: str, value: float) -> None:
        ...

    def send_stats_metrics(self, response: Response, key: str, time_delta: float = None) -> None:
        ...

class BaseSupersetApi(BaseSupersetApiMixin, BaseApi):
    pass

class BaseSupersetModelRestApi(BaseSupersetApiMixin, ModelRestApi):
    method_permission_name: Dict[str, str] = {'bulk_delete': 'delete', 'data': 'list', 'data_from_cache': 'list', 'delete': 'delete', 'distinct': 'list', 'export': 'mulexport', 'import_': 'add', 'get': 'show', 'get_list': 'list', 'info': 'list', 'post': 'add', 'put': 'edit', 'refresh': 'edit', 'related': 'list', 'related_objects': 'list', 'schemas': 'list', 'select_star': 'list', 'table_metadata': 'list', 'test_connection': 'post', 'thumbnail': 'list', 'viz_types': 'list'}
    order_rel_fields: Dict[str, Tuple[str, str]] = {}
    base_related_field_filters: Dict[str, str] = {}
    related_field_filters: Dict[str, RelatedFieldFilter] = {}
    allowed_rel_fields: set = set()
    text_field_rel_fields: Dict[str, str] = {}
    extra_fields_rel_fields: Dict[str, List[str]] = {}
    allowed_distinct_fields: set = set()

    def __init__(self) -> None:
        ...

    def _init_properties(self) -> None:
        ...

    def _get_related_filter(self, datamodel: SQLAInterface, column_name: str, value: str) -> Filters:
        ...

    def _get_distinct_filter(self, column_name: str, value: str) -> Filters:
        ...

    def _get_text_for_model(self, model: Any, column_name: str) -> str:
        ...

    def _get_extra_field_for_model(self, model: Any, column_name: str) -> Dict[str, Any]:
        ...

    def _get_result_from_rows(self, datamodel: SQLAInterface, rows: List[Any], column_name: str) -> List[Dict[str, Any]]:
        ...

    def _add_extra_ids_to_result(self, datamodel: SQLAInterface, column_name: str, ids: List[int], result: List[Dict[str, Any]]) -> None:
        ...

    def info_headless(self, **kwargs) -> Response:
        ...

    def get_headless(self, pk: int, **kwargs) -> Response:
        ...

    def get_list_headless(self, **kwargs) -> Response:
        ...

    def post_headless(self) -> Response:
        ...

    def put_headless(self, pk: int) -> Response:
        ...

    def delete_headless(self, pk: int) -> Response:
        ...

    def related(self, column_name: str, **kwargs) -> Response:
        ...

    def distinct(self, column_name: str, **kwargs) -> Response:
        ...
