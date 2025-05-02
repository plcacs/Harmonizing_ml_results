from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union, cast
from flask import Response
from flask_appbuilder import Model
from flask_appbuilder.api import BaseApi
from flask_appbuilder.models.filters import BaseFilter, Filters
from flask_appbuilder.models.sqla.interface import SQLAInterface
from marshmallow import Schema
from sqlalchemy.orm.query import Query

class RelatedResultResponseSchema(Schema):
    value: int = fields.Integer(metadata={'description': 'The related item identifier'})
    text: str = fields.String(metadata={'description': 'The related item string representation'})
    extra: Dict[str, str] = fields.Dict(metadata={'description': 'The extra metadata for related item'})

class RelatedResponseSchema(Schema):
    count: int = fields.Integer(metadata={'description': 'The total number of related values'})
    result: List[RelatedResultResponseSchema] = fields.List(fields.Nested(RelatedResultResponseSchema))

class DistinctResultResponseSchema(Schema):
    text: str = fields.String(metadata={'description': 'The distinct item'})

class DistincResponseSchema(Schema):
    count: int = fields.Integer(metadata={'description': 'The total number of distinct values'})
    result: List[DistinctResultResponseSchema] = fields.List(fields.Nested(DistinctResultResponseSchema))

class RelatedFieldFilter:

    def __init__(self, field_name, filter_class):
        self.field_name: str = field_name
        self.filter_class: Type[BaseFilter] = filter_class

class BaseSupersetApiMixin:
    csrf_exempt: bool = False
    responses: Dict[str, Dict[str, str]] = {'400': {'description': 'Bad request', 'content': error_payload_content}, '401': {'description': 'Unauthorized', 'content': error_payload_content}, '403': {'description': 'Forbidden', 'content': error_payload_content}, '404': {'description': 'Not found', 'content': error_payload_content}, '410': {'description': 'Gone', 'content': error_payload_content}, '422': {'description': 'Could not process entity', 'content': error_payload_content}, '500': {'description': 'Fatal error', 'content': error_payload_content}}

    def incr_stats(self, action, func_name):
        stats_logger_manager.instance.incr(f'{self.__class__.__name__}.{func_name}.{action}')

    def timing_stats(self, action, func_name, value):
        stats_logger_manager.instance.timing(f'{self.__class__.__name__}.{func_name}.{action}', value)

    def send_stats_metrics(self, response, key, time_delta=None):
        if 200 <= response.status_code < 400:
            self.incr_stats('success', key)
        elif 400 <= response.status_code < 500:
            self.incr_stats('warning', key)
        else:
            self.incr_stats('error', key)
        if time_delta:
            self.timing_stats('time', key, time_delta)

class BaseSupersetApi(BaseSupersetApiMixin, BaseApi):
    pass

class BaseSupersetModelRestApi(BaseSupersetApiMixin, ModelRestApi):
    method_permission_name: Dict[str, str] = {'bulk_delete': 'delete', 'data': 'list', 'data_from_cache': 'list', 'delete': 'delete', 'distinct': 'list', 'export': 'mulexport', 'import_': 'add', 'get': 'show', 'get_list': 'list', 'info': 'list', 'post': 'add', 'put': 'edit', 'refresh': 'edit', 'related': 'list', 'related_objects': 'list', 'schemas': 'list', 'select_star': 'list', 'table_metadata': 'list', 'test_connection': 'post', 'thumbnail': 'list', 'viz_types': 'list'}
    order_rel_fields: Dict[str, Tuple[str, str]] = {}
    base_related_field_filters: Dict[str, BaseFilter] = {}
    related_field_filters: Dict[str, Union[RelatedFieldFilter, str]] = {}
    allowed_rel_fields: Set[str] = set()
    text_field_rel_fields: Dict[str, str] = {}
    extra_fields_rel_fields: Dict[str, List[str]] = {'owners': ['email', 'active']}
    allowed_distinct_fields: Set[str] = set()
    add_columns: List[str]
    edit_columns: List[str]
    list_columns: List[str]
    show_columns: List[str]

    def __init__(self):
        super().__init__()
        if self.apispec_parameter_schemas is None:
            self.apispec_parameter_schemas = {}
        self.apispec_parameter_schemas['get_related_schema'] = get_related_schema
        self.openapi_spec_component_schemas: Tuple[Type[Schema], ...] = self.openapi_spec_component_schemas + (RelatedResponseSchema, DistincResponseSchema)

    def _init_properties(self):
        model_id = self.datamodel.get_pk_name()
        if self.list_columns is None and (not self.list_model_schema):
            self.list_columns = [model_id]
        if self.show_columns is None and (not self.show_model_schema):
            self.show_columns = [model_id]
        if self.edit_columns is None and (not self.edit_model_schema):
            self.edit_columns = [model_id]
        if self.add_columns is None and (not self.add_model_schema):
            self.add_columns = [model_id]
        super()._init_properties()

    def _get_related_filter(self, datamodel, column_name, value):
        filter_field = self.related_field_filters.get(column_name)
        if isinstance(filter_field, str):
            filter_field = RelatedFieldFilter(cast(str, filter_field), FilterStartsWith)
        filter_field = cast(RelatedFieldFilter, filter_field)
        search_columns = [filter_field.field_name] if filter_field else None
        filters = datamodel.get_filters(search_columns)
        if (base_filters := self.base_related_field_filters.get(column_name)):
            filters.add_filter_list(base_filters)
        if value and filter_field:
            filters.add_filter(filter_field.field_name, filter_field.filter_class, value)
        return filters

    def _get_distinct_filter(self, column_name, value):
        filter_field = RelatedFieldFilter(column_name, FilterStartsWith)
        filter_field = cast(RelatedFieldFilter, filter_field)
        search_columns = [filter_field.field_name] if filter_field else None
        filters = self.datamodel.get_filters(search_columns)
        filters.add_filter_list(self.base_filters)
        if value and filter_field:
            filters.add_filter(filter_field.field_name, filter_field.filter_class, value)
        return filters

    def _get_text_for_model(self, model, column_name):
        if column_name in self.text_field_rel_fields:
            model_column_name = self.text_field_rel_fields.get(column_name)
            if model_column_name:
                return getattr(model, model_column_name)
        return str(model)

    def _get_extra_field_for_model(self, model, column_name):
        ret = {}
        if column_name in self.extra_fields_rel_fields:
            model_column_names = self.extra_fields_rel_fields.get(column_name)
            if model_column_names:
                for key in model_column_names:
                    ret[key] = getattr(model, key)
        return ret

    def _get_result_from_rows(self, datamodel, rows, column_name):
        return [{'value': datamodel.get_pk_value(row), 'text': self._get_text_for_model(row, column_name), 'extra': self._get_extra_field_for_model(row, column_name)} for row in rows]

    def _add_extra_ids_to_result(self, datamodel, column_name, ids, result):
        if ids:
            values = [row['value'] for row in result]
            ids = [id_ for id_ in ids if id_ not in values]
            pk_col = datamodel.get_pk()
            extra_rows = db.session.query(datamodel.obj).filter(pk_col.in_(ids)).all()
            result += self._get_result_from_rows(datamodel, extra_rows, column_name)

    @event_logger.log_this_with_context(action=lambda self, *args, **kwargs: f'{self.__class__.__name__}.info', object_ref=False, log_to_statsd=False)
    @handle_api_exception
    def info_headless(self, **kwargs: Any):
        duration, response = time_function(super().info_headless, **kwargs)
        self.send_stats_metrics(response, self.info.__name__, duration)
        return response

    @event_logger.log_this_with_context(action=lambda self, *args, **kwargs: f'{self.__class__.__name__}.get', object_ref=False, log_to_statsd=False)
    @handle_api_exception
    def get_headless(self, pk, **kwargs: Any):
        duration, response = time_function(super().get_headless, pk, **kwargs)
        self.send_stats_metrics(response, self.get.__name__, duration)
        return response

    @event_logger.log_this_with_context(action=lambda self, *args, **kwargs: f'{self.__class__.__name__}.get_list', object_ref=False, log_to_statsd=False)
    @handle_api_exception
    def get_list_headless(self, **kwargs: Any):
        duration, response = time_function(super().get_list_headless, **kwargs)
        self.send_stats_metrics(response, self.get_list.__name__, duration)
        return response

    @event_logger.log_this_with_context(action=lambda self, *args, **kwargs: f'{self.__class__.__name__}.post', object_ref=False, log_to_statsd=False)
    @handle_api_exception
    def post_headless(self):
        duration, response = time_function(super().post_headless)
        self.send_stats_metrics(response, self.post.__name__, duration)
        return response

    @event_logger.log_this_with_context(action=lambda self, *args, **kwargs: f'{self.__class__.__name__}.put', object_ref=False, log_to_statsd=False)
    @handle_api_exception
    def put_headless(self, pk):
        duration, response = time_function(super().put_headless, pk)
        self.send_stats_metrics(response, self.put.__name__, duration)
        return response

    @event_logger.log_this_with_context(action=lambda self, *args, **kwargs: f'{self.__class__.__name__}.delete', object_ref=False, log_to_statsd=False)
    @handle_api_exception
    def delete_headless(self, pk):
        duration, response = time_function(super().delete_headless, pk)
        self.send_stats_metrics(response, self.delete.__name__, duration)
        return response

    @expose('/related/<column_name>', methods=('GET',))
    @protect()
    @safe
    @statsd_metrics
    @rison(get_related_schema)
    @handle_api_exception
    def related(self, column_name, **kwargs: Any):
        if column_name not in self.allowed_rel_fields:
            self.incr_stats('error', self.related.__name__)
            return self.response_404()
        args = kwargs.get('rison', {})
        page, page_size = self._handle_page_args(args)
        ids = args.get('include_ids')
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
        filters = self._get_related_filter(datamodel, column_name, args.get('filter'))
        total_rows, rows = datamodel.query(filters, order_column, order_direction, page=page, page_size=page_size)
        result = self._get_result_from_rows(datamodel, rows, column_name)
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
    def distinct(self, column_name, **kwargs: Any):
        if column_name not in self.allowed_distinct_fields:
            self.incr_stats('error', self.related.__name__)
            return self.response_404()
        args = kwargs.get('rison', {})
        page, page_size = self._sanitize_page_args(*self._handle_page_args(args))
        filters = self._get_distinct_filter(column_name, args.get('filter'))
        query_count = self.appbuilder.get_session.query(func.count(distinct(getattr(self.datamodel.obj, column_name))))
        count = self.datamodel.apply_filters(query_count, filters).scalar()
        if count == 0:
            return self.response(200, count=count, result=[])
        query = self.appbuilder.get_session.query(distinct(getattr(self.datamodel.obj, column_name)))
        query = self.datamodel.apply_filters(query, filters)
        query = self.datamodel.apply_order_by(query, column_name, 'asc')
        result = self.datamodel.apply_pagination(query, page, page_size).all()
        result = [{'text': item[0], 'value': item[0]} for item in result if item[0] is not None]
        return self.response(200, count=count, result=result)