from typing import Any, Optional, Union, List, Dict, Tuple, Set, cast
from datetime import datetime, timedelta
from collections.abc import Hashable
import sqlalchemy as sa
from sqlalchemy.sql.elements import ColumnElement, TextClause
from sqlalchemy.sql.selectable import Alias, TableClause
from sqlalchemy.orm import Mapper
from flask import g
from flask_appbuilder import Model
from flask_appbuilder.models.mixins import AuditMixin
from flask_appbuilder.security.sqla.models import User
from flask_babel import lazy_gettext as _
from markupsafe import escape, Markup
from sqlalchemy import and_, Column, or_, UniqueConstraint
from sqlalchemy.exc import MultipleResultsFound
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import validates
from sqlalchemy_utils import UUIDType
import pandas as pd
import numpy as np
import json
import re
import uuid
import logging
import dateutil.parser
import humanize
import pytz
import sqlparse
import yaml
from superset import app, db, is_feature_enabled
from superset.advanced_data_type.types import AdvancedDataTypeResponse
from superset.common.db_query_status import QueryStatus
from superset.common.utils.time_range_utils import get_since_until_from_time_range
from superset.constants import EMPTY_STRING, NULL_STRING
from superset.db_engine_specs.base import TimestampExpression
from superset.errors import ErrorLevel, SupersetError, SupersetErrorType
from superset.exceptions import AdvancedDataTypeResponseError, ColumnNotFoundException, QueryClauseValidationException, QueryObjectValidationError, SupersetParseError, SupersetSecurityException
from superset.extensions import feature_flag_manager
from superset.jinja_context import BaseTemplateProcessor
from superset.sql.parse import SQLScript
from superset.sql_parse import has_table_query, insert_rls_in_predicate, sanitize_clause
from superset.superset_typing import AdhocMetric, Column as ColumnTyping, FilterValue, FilterValues, Metric, OrderBy, QueryObjectDict
from superset.utils import core as utils
from superset.utils.core import GenericDataType, get_column_name, get_non_base_axis_columns, get_user_id, is_adhoc_column, MediumText, remove_duplicates
from superset.utils.dates import datetime_to_epoch
if TYPE_CHECKING:
    from superset.connectors.sqla.models import SqlMetric, TableColumn
    from superset.db_engine_specs import BaseEngineSpec
    from superset.models.core import Database
config = app.config
logger = logging.getLogger(__name__)
VIRTUAL_TABLE_ALIAS = 'virtual_table'
SERIES_LIMIT_SUBQ_ALIAS = 'series_limit'
ADVANCED_DATA_TYPES = config['ADVANCED_DATA_TYPES']

def validate_adhoc_subquery(sql, database_id, engine, default_schema):
    statements = []
    for statement in sqlparse.parse(sql):
        try:
            has_table = has_table_query(str(statement), engine)
        except SupersetParseError:
            has_table = True
        if has_table:
            if not is_feature_enabled('ALLOW_ADHOC_SUBQUERY'):
                raise SupersetSecurityException(SupersetError(error_type=SupersetErrorType.ADHOC_SUBQUERY_NOT_ALLOWED_ERROR, message=_('Custom SQL fields cannot contain sub-queries.'), level=ErrorLevel.ERROR))
            statement = insert_rls_in_predicate(statement, database_id, default_schema)
        statements.append(statement)
    return ';\n'.join((str(statement) for statement in statements))

def json_to_dict(json_str):
    if json_str:
        val = re.sub(',[ \t\r\n]+}', '}', json_str)
        val = re.sub(',[ \t\r\n]+\\]', ']', val)
        return json.loads(val)
    return {}

def convert_uuids(obj):
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, list):
        return [convert_uuids(el) for el in obj]
    if isinstance(obj, dict):
        return {k: convert_uuids(v) for k, v in obj.items()}
    return obj

class UUIDMixin:
    uuid = sa.Column(UUIDType(binary=True), primary_key=False, unique=True, default=uuid.uuid4)

    @property
    def short_uuid(self):
        return str(self.uuid)[:8]

class ImportExportMixin(UUIDMixin):
    export_parent: Optional[str] = None
    export_children: List[str] = []
    export_fields: List[str] = []
    extra_import_fields: List[str] = []
    __mapper__: Mapper

    @classmethod
    def _unique_constraints(cls):
        unique = [{c.name for c in u.columns} for u in cls.__table_args__ if isinstance(u, UniqueConstraint)]
        unique.extend(({c.name} for c in cls.__table__.columns if c.unique))
        return unique

    @classmethod
    def parent_foreign_key_mappings(cls):
        parent_rel = cls.__mapper__.relationships.get(cls.export_parent)
        if parent_rel:
            return {local.name: remote.name for local, remote in parent_rel.local_remote_pairs}
        return {}

    @classmethod
    def export_schema(cls, recursive=True, include_parent_ref=False):
        parent_excludes = set()
        if not include_parent_ref:
            parent_ref = cls.__mapper__.relationships.get(cls.export_parent)
            if parent_ref:
                parent_excludes = {column.name for column in parent_ref.local_columns}

        def formatter(column):
            return f'{str(column.type)} Default ({column.default.arg})' if column.default else str(column.type)
        schema: Dict[str, Any] = {column.name: formatter(column) for column in cls.__table__.columns if column.name in cls.export_fields and column.name not in parent_excludes}
        if recursive:
            for column in cls.export_children:
                child_class = cls.__mapper__.relationships[column].argument.class_
                schema[column] = [child_class.export_schema(recursive=recursive, include_parent_ref=include_parent_ref)]
        return schema

    @classmethod
    def import_from_dict(cls, dict_rep, parent=None, recursive=True, sync=None, allow_reparenting=False):
        if sync is None:
            sync = []
        parent_refs = cls.parent_foreign_key_mappings()
        export_fields = set(cls.export_fields) | set(cls.extra_import_fields) | set(parent_refs.keys()) | {'uuid'}
        new_children = {c: dict_rep[c] for c in cls.export_children if c in dict_rep}
        unique_constraints = cls._unique_constraints()
        filters = []
        for k in list(dict_rep):
            if k not in export_fields and k not in parent_refs:
                del dict_rep[k]
        if not parent:
            if cls.export_parent:
                for prnt in parent_refs.keys():
                    if prnt not in dict_rep:
                        raise RuntimeError(f'{cls.__name__}: Missing field {prnt}')
        else:
            for k, v in parent_refs.items():
                dict_rep[k] = getattr(parent, v)
        if not allow_reparenting:
            filters.extend([getattr(cls, k) == dict_rep.get(k) for k in parent_refs.keys()])
        ucs = [and_(*[getattr(cls, k) == dict_rep.get(k) for k in cs if dict_rep.get(k) is not None]) for cs in unique_constraints]
        filters.append(or_(*ucs))
        try:
            obj_query = db.session.query(cls).filter(and_(*filters))
            obj = obj_query.one_or_none()
        except MultipleResultsFound:
            logger.error('Error importing %s \n %s \n %s', cls.__name__, str(obj_query), yaml.safe_dump(dict_rep), exc_info=True)
            raise
        if not obj:
            is_new_obj = True
            obj = cls(**dict_rep)
            logger.debug('Importing new %s %s', obj.__tablename__, str(obj))
            if cls.export_parent and parent:
                setattr(obj, cls.export_parent, parent)
            db.session.add(obj)
        else:
            is_new_obj = False
            logger.debug('Updating %s %s', obj.__tablename__, str(obj))
            for k, v in dict_rep.items():
                setattr(obj, k, v)
        if recursive:
            for child in cls.export_children:
                argument = cls.__mapper__.relationships[child].argument
                child_class = argument.class_ if hasattr(argument, 'class_') else argument
                added = []
                for c_obj in new_children.get(child, []):
                    added.append(child_class.import_from_dict(dict_rep=c_obj, parent=obj, sync=sync))
                if child in sync and (not is_new_obj):
                    back_refs = child_class.parent_foreign_key_mappings()
                    delete_filters = [getattr(child_class, k) == getattr(obj, back_refs.get(k)) for k in back_refs.keys()]
                    to_delete = set(db.session.query(child_class).filter(and_(*delete_filters))).difference(set(added))
                    for o in to_delete:
                        logger.debug('Deleting %s %s', child, str(obj))
                        db.session.delete(o)
        return obj

    def export_to_dict(self, recursive=True, include_parent_ref=False, include_defaults=False, export_uuids=False):
        export_fields = set(self.export_fields)
        if export_uuids:
            export_fields.add('uuid')
            if 'id' in export_fields:
                export_fields.remove('id')
        cls = self.__class__
        parent_excludes = set()
        if recursive and (not include_parent_ref):
            parent_ref = cls.__mapper__.relationships.get(cls.export_parent)
            if parent_ref:
                parent_excludes = {c.name for c in parent_ref.local_columns}
        dict_rep = {c.name: getattr(self, c.name) for c in cls.__table__.columns if c.name in export_fields and c.name not in parent_excludes and (include_defaults or (getattr(self, c.name) is not None and (not c.default or getattr(self, c.name) != c.default.arg)))}
        order = {field: i for i, field in enumerate(self.export_fields)}
        decorated_keys = [(order.get(k, len(order)), k) for k in dict_rep]
        decorated_keys.sort()
        dict_rep = {k: dict_rep[k] for _, k in decorated_keys}
        if recursive:
            for cld in self.export_children:
                dict_rep[cld] = sorted([child.export_to_dict(recursive=recursive, include_parent_ref=include_parent_ref, include_defaults=include_defaults) for child in getattr(self, cld)], key=lambda k: sorted(str(k.items())))
        return convert_uuids(dict_rep)

    def override(self, obj):
        for field in obj.__class__.export_fields:
            setattr(self, field, getattr(obj, field))

    def copy(self):
        new_obj = self.__class__()
        new_obj.override(self)
        return new_obj

    def alter_params(self, **kwargs: Any):
        params = self.params_dict
        params.update(kwargs)
        self.params = json.dumps(params)

    def remove_params(self, param_to_remove):
        params = self.params_dict
        params.pop(param_to_remove, None)
        self.params = json.dumps(params)

    def reset_ownership(self):
        self.created_by = None
        self.changed_by = None
        self.owners = []
        if g and hasattr(g, 'user'):
            self.owners = [g.user]

    @property
    def params_dict(self):
        return json_to_dict(self.params)

    @property
    def template_params_dict(self):
        return json_to_dict(self.template_params)

def _user(user):
    if not user:
        return ''
    return escape(user)

class AuditMixinNullable(AuditMixin):
    created_on = sa.Column(sa.DateTime, default=datetime.now, nullable=True)
    changed_on = sa.Column(sa.DateTime, default=datetime.now, onupdate=datetime.now, nullable=True)

    @declared_attr
    def created_by_fk(self):
        return sa.Column(sa.Integer, sa.ForeignKey('ab_user.id'), default=get_user_id, nullable=True)

    @declared_attr
    def changed_by_fk(self):
        return sa.Column(sa.Integer, sa.ForeignKey('ab_user.id'), default=get_user_id, onupdate=get_user_id, nullable=True)

    @property
    def created_by_name(self):
        if self.created_by:
            return escape(f'{self.created_by}')
        return ''

    @property
    def changed_by_name(self):
        if self.changed_by:
            return escape(f'{self.changed_by}')
        return ''

    @renders('created_by')
    def creator(self):
        return _user(self.created_by)

    @property
    def changed_by_(self):
        return _user(self.changed_by)

    @renders('changed_on')
    def changed_on_(self):
        return Markup(f'<span class="no-wrap">{self.changed_on}</span>')

    @renders('changed_on')
    def changed_on_delta_humanized(self):
        return self.changed_on_humanized

    @renders('changed_on')
    def changed_on_dttm(self):
        return datetime_to_epoch(self.changed_on)

    @renders('created_on')
    def created_on_delta_humanized(self):
        return self.created_on_humanized

    @renders('changed_on')
    def changed_on_utc(self):
        return self.changed_on.astimezone(pytz.utc).strftime('%Y-%m-%dT%H:%M:%S.%f%z')

    @property
    def changed_on_humanized(self):
        return humanize.naturaltime(datetime.now() - self.changed_on)

    @property
    def created_on_humanized(self):
        return humanize.naturaltime(datetime.now() - self.created_on)

    @renders('changed_on')
    def modified(self):
        return Markup(f'<span class="no-wrap">{self.changed_on_humanized}</span>')

class QueryResult:

    def __init__(self, df, query, duration, applied_template_filters=None, applied_filter_columns=None, rejected_filter_columns=None, status=QueryStatus.SUCCESS, error_message=None, errors=None, from_dttm=None, to_dttm=None):
        self.df = df
        self.query = query
        self.duration = duration
        self.applied_template_filters = applied_template_filters or []
        self.applied_filter_columns = applied_filter_columns or []
        self.rejected_filter_columns = rejected_filter_columns or []
        self.status = status
        self.error_message = error_message
        self.errors = errors or []
        self.from_dttm = from_dttm
        self.to_dttm = to_dttm
        self.sql_rowcount = len(self.df.index) if not self.df.empty else 0

class ExtraJSONMixin:
    extra_json = sa.Column(MediumText(), default='{}')

    @property
    def extra(self):
        try:
            return json.loads(self.extra_json or '{}') or {}
        except (TypeError, json.JSONDecodeError) as exc:
            logger.error('Unable to load an extra json: %r. Leaving empty.', exc, exc_info=True)
            return {}

    @extra.setter
    def extra(self, extras):
        self.extra_json = json.dumps(extras)

    def set_extra_json_key(self, key, value):
        extra = self.extra
        extra[key] = value
        self.extra_json = json.dumps(extra)

    @validates('extra_json')
    def ensure_extra_json_is_not_none(self, _, value):
        if value is None:
            return '{}'
        return value

class CertificationMixin:
    extra = sa.Column(sa.Text, default='{}')

    def get_extra_dict(self):
        try:
            return json.loads(self.extra)
        except (TypeError, json.JSONDecodeError):
            return {}

    @property
    def is_certified(self):
        return bool(self.get_extra_dict().get('certification'))

    @property
    def certified_by(self):
        return self.get_extra_dict().get('certification', {}).get('certified_by')

    @property
    def certification_details(self):
        return self.get_extra_dict().get('certification', {}).get('details')

    @property
    def warning_markdown(self):
        return self.get_extra_dict().get('warning_markdown')

def clone_model(target, ignore=None, keep_relations=None, **kwargs: Any):
    ignore = ignore or []
    table = target.__table__
    primary_keys = table.primary_key.columns.keys()
    data = {attr: getattr(target, attr) for attr in list(table.columns.keys()) + (keep_relations or []) if attr not in primary_keys and attr not in ignore}
    data.update(kwargs)
    return target.__class__(**data)

class QueryStringExtended(NamedTuple):
    applied_template_filters: Optional[List[str]]
    applied_filter_columns: List[ColumnTyping]
    rejected_filter_columns: List[ColumnTyping]
    labels_expected: List[str]
    prequeries: List[str]
    sql: str

class SqlaQuery(NamedTuple):
    applied_template_filters: List[str]
    applied_filter_columns: List[ColumnTyping]
    rejected_filter_columns: List[ColumnTyping]
    cte: Optional[str]
    extra_cache_keys: List[Any]
    labels_expected: List[str]
    prequeries: List[str]
    sqla_query: Select

class ExploreMixin:
    sqla_aggregations = {'COUNT_DISTINCT': lambda column_name: sa.func.COUNT(sa.distinct(column_name)), 'COUNT': sa.func.COUNT, 'SUM': sa.func.SUM, 'AVG': sa.func.AVG, 'MIN': sa.func.MIN, 'MAX': sa.func.MAX}
    fetch_values_predicate = None

    @property
    def type(self):
        raise NotImplementedError()

    @property
    def db_extra(self):
        raise NotImplementedError()

    def query(self, query_obj):
        raise NotImplementedError()

    @property
    def database_id(self):
        raise NotImplementedError()

    @property
    def owners_data(self):
        raise NotImplementedError()

    @property
    def metrics(self):
        return []

    @property
    def uid(self):
        raise NotImplementedError()

    @property
    def is_rls_supported(self):
        raise NotImplementedError()

    @property
    def cache_timeout(self):
        raise NotImplementedError()

    @property
    def column_names(self):
        raise NotImplementedError()

    @property
    def offset(self):
        raise NotImplementedError()

    @property
    def main_dttm_col(self):
        raise NotImplementedError()

    @property
    def always_filter_main_dttm(self):
        return False

    @property
    def dttm_cols(self):
        raise NotImplementedError()

    @property
    def db_engine_spec(self):
        raise NotImplementedError()

    @property
    def database(self):
        raise NotImplementedError()

    @property
    def catalog(self):
        raise NotImplementedError()

    @property
    def schema(self):
        raise NotImplementedError()

    @property
    def sql(self):
        raise NotImplementedError()

    @property
    def columns(self):
        raise NotImplementedError()

    def get_extra_cache_keys(self, query_obj):
        raise NotImplementedError()

    def get_template_processor(self, **kwargs: Any):
        raise NotImplementedError()

    def get_fetch_values_predicate(self, template_processor=None):
        return self.fetch_values_predicate

    def get_sqla_row_level_filters(self, template_processor=None):
        return []

    def _process_sql_expression(self, expression, database_id, engine, schema, template_processor):
        if template_processor and expression:
            expression = template_processor.process_template(expression)
        if expression:
            expression = validate_adhoc_subquery(expression, database_id, engine, schema)
            try:
                expression = sanitize_clause(expression)
            except QueryClauseValidationException as ex:
                raise QueryObjectValidationError(ex.message) from ex
        return expression

    def make_sqla_column_compatible(self, sqla_col, label=None):
        label_expected = label or sqla_col.name
        db_engine_spec = self.db_engine_spec
        if db_engine_spec.get_allows_alias_in_select(self.database):
            label = db_engine_spec.make_label_compatible(label_expected)
            sqla_col = sqla_col.label(label)
        sqla_col.key = label_expected
        return sqla_col

    @staticmethod
    def _apply_cte(sql, cte):
        if cte:
            sql = f'{cte}\n{sql}'
        return sql

    def get_query_str_extended(self, query_obj, mutate=True):
        sqlaq = self.get_sqla_query(**query_obj)
        sql = self.database.compile_sqla_query(sqlaq.sqla_query, catalog=self.catalog, schema=self.schema, is_virtual=bool(self.sql))
        sql = self._apply_cte(sql, sqlaq.cte)
        if mutate:
            sql = self.database.mutate_sql_based_on_config(sql)
        return QueryStringExtended(applied_template_filters=sqlaq.applied_template_filters, applied_filter_columns=sqlaq.applied_filter_columns, rejected_filter_columns=sqlaq.rejected_filter_columns, labels_expected=sqlaq.labels_expected, prequeries=sqlaq.prequeries, sql=sql)

    def _normalize_prequery_result_type(self, row, dimension, columns_by_name):
        value = row[dimension]
        if isinstance(value, np.generic):
            value = value.item()
        column_ = columns_by_name[dimension]
        db_extra: Dict[str, Any] = self.database.get_extra()
        if isinstance(column_, dict):
            if column_.get('type') and column_.get('is_temporal') and isinstance(value, str):
                sql = self.db_engine_spec.convert_dttm(column_.get('type'), dateutil.parser.parse(value), db_extra=None)
                if sql:
                    value = self.db_engine_spec.get_text_clause(sql)
        elif column_.type and column_.is_temporal and isinstance(value, str):
            sql = self.db_engine_spec.convert_dttm(column_.type, dateutil.parser.parse(value), db_extra=db_extra)
            if sql:
                value = self.text(sql)
        return value

    def make_orderby_compatible(self, select_exprs, orderby_exprs):

        def is_alias_used_in_orderby(col):
            if not isinstance(col, Label):
                return False
            regexp = re.compile(f'\\(.*\\b{re.escape(col.name)}\\b.*\\)', re.IGNORECASE)
            return any((regexp.search(str(x)) for x in orderby_exprs))
        for col in select_exprs:
            if is_alias_used_in_orderby(col):
                col.name = f'{col.name}__'

    def exc_query(self, qry):
        qry_start_dttm = datetime.now()
        query_str_ext = self.get_query_str_extended(qry)
        sql = query_str_ext.sql
        status = QueryStatus.SUCCESS
        errors = None
        error_message = None

        def assign_column_label(df):
            labels_expected = query_str_ext.labels_expected
            if df is not None and (not df.empty):
                if len(df.columns) < len(labels_expected):
                    raise QueryObjectValidationError(_('Db engine did not return all queried columns'))
                if len(df.columns) > len(labels_expected):
                    df = df.iloc[:, 0:len(labels_expected)]
                df.columns = labels_expected
            return df
        try:
            df = self.database.get_df(sql, self.catalog, self.schema, mutator=assign_column_label)
        except Exception as ex:
            df = pd.DataFrame()
            status = QueryStatus.FAILED
            logger.warning('Query %s on schema %s failed', sql, self.schema, exc_info=True)
            db_engine_spec = self.db_engine_spec
            errors = [dataclasses.asdict(error) for error in db_engine_spec.extract_errors(ex)]
            error_message = utils.error_msg_from_exception(ex)
        return QueryResult(applied_template_filters=query_str_ext.applied_template_filters, applied_filter_columns=query_str_ext.applied_filter_columns, rejected_filter_columns=query_str_ext.rejected_filter_columns, status=status, df=df, duration=datetime.now() - qry_start_dttm, query=sql, errors=errors, error_message=error_message)

    def get_rendered_sql(self, template_processor=None):
        if not self.sql:
            return ''
        sql = self.sql.strip('\t\r\n; ')
        if template_processor:
            try:
                sql = template_processor.process_template(sql)
            except TemplateError as ex:
                raise QueryObjectValidationError(_('Error while rendering virtual dataset query: %(msg)s', msg=ex.message)) from ex
        script = SQLScript(sql, engine=self.db_engine_spec.engine)
        if len(script.statements) > 1:
            raise QueryObjectValidationError(_('Virtual dataset query cannot consist of multiple statements'))
        if not sql:
            raise QueryObjectValidationError(_('Virtual dataset query cannot be empty'))
        return sql

    def text(self, clause):
        return self.db_engine_spec.get_text_clause(clause)

    def get_from_clause(self, template_processor=None):
        from_sql = self.get_rendered_sql(template_processor) + '\n'
        parsed_script = SQLScript(from_sql, engine=self.db_engine_spec.engine)
        if parsed_script.has_mutation():
            raise QueryObjectValidationError(_('Virtual dataset query must be read-only'))
        cte = self.db_engine_spec.get_cte_query(from_sql)
        from_clause = sa.table(self.db_engine_spec.cte_alias) if cte else TextAsFrom(self.text(from_sql), []).alias(VIRTUAL_TABLE_ALIAS)
        return (from_clause, cte)

    def adhoc_metric_to_sqla(self, metric, columns_by_name, template_processor=None):
        expression_type = metric.get('expressionType')
        label = utils.get_metric_name(metric)
        if expression_type == utils.AdhocMetricExpressionType.SIMPLE:
            metric_column = metric.get('column') or {}
            column_name = cast(str, metric_column.get('column_name'))
            sqla_column = sa.column(column_name)
            sqla_metric = self.sqla_aggregations[metric['aggregate']](sqla_column)
        elif expression_type == utils.AdhocMetricExpressionType.SQL:
            expression = self._process_sql_expression(expression=metric['sqlExpression'], database_id=self.database_id, engine=self.database.backend, schema=self.schema, template_processor=template_processor)
            sqla_metric = literal_column(expression)
        else:
            raise QueryObjectValidationError('Adhoc metric expressionType is invalid')
        return self.make_sqla_column_compatible(sqla_metric, label)

    @property
    def template_params_dict(self):
        return {}

    @staticmethod
    def filter_values_handler(values, operator, target_generic_type, target_native_type=None, is_list_target=False, db_engine_spec=None, db_extra=None):
        if values is None:
            return None

        def handle_single_value(value):
            if operator == utils.FilterOperator.TEMPORAL_RANGE:
                return value
            if isinstance(value, (float, int)) and target_generic_type == utils.GenericDataType.TEMPORAL and (target_native_type is not None) and (db_engine_spec is not None):
                value = db_engine_spec.convert_dttm(target_type=target_native_type, dttm=datetime.utcfromtimestamp(value / 1000), db_extra=db_extra)
                value = literal_column(value)
            if isinstance(value, str):
                value = value.strip('\t\n')
                if target_generic_type == utils.GenericDataType.NUMERIC and operator not in {utils.FilterOperator.ILIKE, utils.FilterOperator.LIKE}:
                    return utils.cast_to_num(value)
                if value == NULL_STRING:
                    return None
                if value == EMPTY_STRING:
                    return ''
            if target_generic_type == utils.GenericDataType.BOOLEAN:
                return utils.cast_to_boolean(value)
            return value
        if isinstance(values, (list, tuple)):
            values = [handle_single_value(v) for v in values]
        else:
            values = handle_single_value(values)
        if is_list_target and (not isinstance(values, (tuple, list))):
            values = [values]
        elif not is_list_target and isinstance(values, (tuple, list)):
            values = values[0] if values else None
        return values

    def get_query_str(self, query_obj):
        query_str_ext = self.get_query_str_extended(query_obj)
        all_queries = query_str_ext.prequeries + [query_str_ext.sql]
        return ';\n\n'.join(all_queries) + ';'

    def _get_series_orderby(self, series_limit_metric, metrics_by_name, columns_by_name, template_processor=None):
        if utils.is_adhoc_metric(series_limit_metric):
            assert isinstance(series_limit_metric, dict)
            ob = self.adhoc_metric_to_sqla(series_limit_metric, columns_by_name)
        elif isinstance(series_limit_metric, str) and series_limit_metric in metrics_by_name:
            ob = metrics_by_name[series_limit_metric].get_sqla_col(template_processor=template_processor)
        else:
            raise QueryObjectValidation