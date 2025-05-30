"""a collection of model-related helper classes and functions"""
import builtins
import dataclasses
import logging
import re
import uuid
from collections.abc import Hashable
from datetime import datetime, timedelta
from typing import Any, cast, NamedTuple, Optional, TYPE_CHECKING, Union
import dateutil.parser
import humanize
import numpy as np
import pandas as pd
import pytz
import sqlalchemy as sa
import sqlparse
import yaml
from flask import g
from flask_appbuilder import Model
from flask_appbuilder.models.decorators import renders
from flask_appbuilder.models.mixins import AuditMixin
from flask_appbuilder.security.sqla.models import User
from flask_babel import lazy_gettext as _
from jinja2.exceptions import TemplateError
from markupsafe import escape, Markup
from sqlalchemy import and_, Column, or_, UniqueConstraint
from sqlalchemy.exc import MultipleResultsFound
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import Mapper, validates
from sqlalchemy.sql.elements import ColumnElement, literal_column, TextClause
from sqlalchemy.sql.expression import Label, Select, TextAsFrom
from sqlalchemy.sql.selectable import Alias, TableClause
from sqlalchemy_utils import UUIDType
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
from superset.utils import core as utils, json
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
    """
    Check if adhoc SQL contains sub-queries or nested sub-queries with table.

    If sub-queries are allowed, the adhoc SQL is modified to insert any applicable RLS
    predicates to it.

    :param sql: adhoc sql expression
    :raise SupersetSecurityException if sql contains sub-queries or
    nested sub-queries with table
    """
    statements = []
    for statement in sqlparse.parse(sql):
        try:
            has_table = has_table_query(str(statement), engine)
        except SupersetParseError:
            has_table = True
        if has_table:
            if not is_feature_enabled('ALLOW_ADHOC_SUBQUERY'):
                raise SupersetSecurityException(SupersetError(error_type=
                    SupersetErrorType.ADHOC_SUBQUERY_NOT_ALLOWED_ERROR,
                    message=_(
                    'Custom SQL fields cannot contain sub-queries.'), level
                    =ErrorLevel.ERROR))
            statement = insert_rls_in_predicate(statement, database_id,
                default_schema)
        statements.append(statement)
    return ';\n'.join(str(statement) for statement in statements)


def json_to_dict(json_str):
    if json_str:
        val = re.sub(',[ \t\r\n]+}', '}', json_str)
        val = re.sub(',[ \t\r\n]+\\]', ']', val)
        return json.loads(val)
    return {}


def convert_uuids(obj):
    """
    Convert UUID objects to str so we can use yaml.safe_dump
    """
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, list):
        return [convert_uuids(el) for el in obj]
    if isinstance(obj, dict):
        return {k: convert_uuids(v) for k, v in obj.items()}
    return obj


class UUIDMixin:
    uuid = sa.Column(UUIDType(binary=True), primary_key=False, unique=True,
        default=uuid.uuid4)

    @property
    def short_uuid(self):
        return str(self.uuid)[:8]


class ImportExportMixin(UUIDMixin):
    export_parent: Optional[str] = None
    export_children: list[str] = []
    export_fields: list[str] = []
    extra_import_fields: list[str] = []
    __mapper__: Mapper

    @classmethod
    def _unique_constraints(cls):
        """Get all (single column and multi column) unique constraints"""
        unique = [{c.name for c in u.columns} for u in cls.__table_args__ if
            isinstance(u, UniqueConstraint)]
        unique.extend({c.name} for c in cls.__table__.columns if c.unique)
        return unique

    @classmethod
    def parent_foreign_key_mappings(cls):
        """Get a mapping of foreign name to the local name of foreign keys"""
        parent_rel = cls.__mapper__.relationships.get(cls.export_parent)
        if parent_rel:
            return {local.name: remote.name for local, remote in parent_rel
                .local_remote_pairs}
        return {}

    @classmethod
    def export_schema(cls, recursive=True, include_parent_ref=False):
        """Export schema as a dictionary"""
        parent_excludes = set()
        if not include_parent_ref:
            parent_ref = cls.__mapper__.relationships.get(cls.export_parent)
            if parent_ref:
                parent_excludes = {column.name for column in parent_ref.
                    local_columns}

        def formatter(column):
            return (f'{str(column.type)} Default ({column.default.arg})' if
                column.default else str(column.type))
        schema: dict[str, Any] = {column.name: formatter(column) for column in
            cls.__table__.columns if column.name in cls.export_fields and 
            column.name not in parent_excludes}
        if recursive:
            for column in cls.export_children:
                child_class = cls.__mapper__.relationships[column
                    ].argument.class_
                schema[column] = [child_class.export_schema(recursive=
                    recursive, include_parent_ref=include_parent_ref)]
        return schema

    @classmethod
    def import_from_dict(cls, dict_rep, parent=None, recursive=True, sync=
        None, allow_reparenting=False):
        """Import obj from a dictionary"""
        if sync is None:
            sync = []
        parent_refs = cls.parent_foreign_key_mappings()
        export_fields = set(cls.export_fields) | set(cls.extra_import_fields
            ) | set(parent_refs.keys()) | {'uuid'}
        new_children = {c: dict_rep[c] for c in cls.export_children if c in
            dict_rep}
        unique_constraints = cls._unique_constraints()
        filters = []
        for k in list(dict_rep):
            if k not in export_fields and k not in parent_refs:
                del dict_rep[k]
        if not parent:
            if cls.export_parent:
                for prnt in parent_refs.keys():
                    if prnt not in dict_rep:
                        raise RuntimeError(
                            f'{cls.__name__}: Missing field {prnt}')
        else:
            for k, v in parent_refs.items():
                dict_rep[k] = getattr(parent, v)
        if not allow_reparenting:
            filters.extend([(getattr(cls, k) == dict_rep.get(k)) for k in
                parent_refs.keys()])
        ucs = [and_(*[(getattr(cls, k) == dict_rep.get(k)) for k in cs if 
            dict_rep.get(k) is not None]) for cs in unique_constraints]
        filters.append(or_(*ucs))
        try:
            obj_query = db.session.query(cls).filter(and_(*filters))
            obj = obj_query.one_or_none()
        except MultipleResultsFound:
            logger.error('Error importing %s \n %s \n %s', cls.__name__,
                str(obj_query), yaml.safe_dump(dict_rep), exc_info=True)
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
                child_class = argument.class_ if hasattr(argument, 'class_'
                    ) else argument
                added = []
                for c_obj in new_children.get(child, []):
                    added.append(child_class.import_from_dict(dict_rep=
                        c_obj, parent=obj, sync=sync))
                if child in sync and not is_new_obj:
                    back_refs = child_class.parent_foreign_key_mappings()
                    delete_filters = [(getattr(child_class, k) == getattr(
                        obj, back_refs.get(k))) for k in back_refs.keys()]
                    to_delete = set(db.session.query(child_class).filter(
                        and_(*delete_filters))).difference(set(added))
                    for o in to_delete:
                        logger.debug('Deleting %s %s', child, str(obj))
                        db.session.delete(o)
        return obj

    def export_to_dict(self, recursive=True, include_parent_ref=False,
        include_defaults=False, export_uuids=False):
        """Export obj to dictionary"""
        export_fields = set(self.export_fields)
        if export_uuids:
            export_fields.add('uuid')
            if 'id' in export_fields:
                export_fields.remove('id')
        cls = self.__class__
        parent_excludes = set()
        if recursive and not include_parent_ref:
            parent_ref = cls.__mapper__.relationships.get(cls.export_parent)
            if parent_ref:
                parent_excludes = {c.name for c in parent_ref.local_columns}
        dict_rep = {c.name: getattr(self, c.name) for c in cls.__table__.
            columns if c.name in export_fields and c.name not in
            parent_excludes and (include_defaults or getattr(self, c.name)
             is not None and (not c.default or getattr(self, c.name) != c.
            default.arg))}
        order = {field: i for i, field in enumerate(self.export_fields)}
        decorated_keys = [(order.get(k, len(order)), k) for k in dict_rep]
        decorated_keys.sort()
        dict_rep = {k: dict_rep[k] for _, k in decorated_keys}
        if recursive:
            for cld in self.export_children:
                dict_rep[cld] = sorted([child.export_to_dict(recursive=
                    recursive, include_parent_ref=include_parent_ref,
                    include_defaults=include_defaults) for child in getattr
                    (self, cld)], key=lambda k: sorted(str(k.items())))
        return convert_uuids(dict_rep)

    def override(self, obj):
        """Overrides the plain fields of the dashboard."""
        for field in obj.__class__.export_fields:
            setattr(self, field, getattr(obj, field))

    def copy(self):
        """Creates a copy of the dashboard without relationships."""
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
        """object will belong to the user the current user"""
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
    """Altering the AuditMixin to use nullable fields

    Allows creating objects programmatically outside of CRUD
    """
    created_on = sa.Column(sa.DateTime, default=datetime.now, nullable=True)
    changed_on = sa.Column(sa.DateTime, default=datetime.now, onupdate=
        datetime.now, nullable=True)

    @declared_attr
    def created_by_fk(self):
        return sa.Column(sa.Integer, sa.ForeignKey('ab_user.id'), default=
            get_user_id, nullable=True)

    @declared_attr
    def changed_by_fk(self):
        return sa.Column(sa.Integer, sa.ForeignKey('ab_user.id'), default=
            get_user_id, onupdate=get_user_id, nullable=True)

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
        return self.changed_on.astimezone(pytz.utc).strftime(
            '%Y-%m-%dT%H:%M:%S.%f%z')

    @property
    def changed_on_humanized(self):
        return humanize.naturaltime(datetime.now() - self.changed_on)

    @property
    def created_on_humanized(self):
        return humanize.naturaltime(datetime.now() - self.created_on)

    @renders('changed_on')
    def modified(self):
        return Markup(
            f'<span class="no-wrap">{self.changed_on_humanized}</span>')


class QueryResult:
    """Object returned by the query interface"""

    def __init__(self, df, query, duration, applied_template_filters=None,
        applied_filter_columns=None, rejected_filter_columns=None, status=
        QueryStatus.SUCCESS, error_message=None, errors=None, from_dttm=
        None, to_dttm=None):
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
    """Mixin to add an `extra` column (JSON) and utility methods"""
    extra_json = sa.Column(MediumText(), default='{}')

    @property
    def extra(self):
        try:
            return json.loads(self.extra_json or '{}') or {}
        except (TypeError, json.JSONDecodeError) as exc:
            logger.error('Unable to load an extra json: %r. Leaving empty.',
                exc, exc_info=True)
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
    """Mixin to add extra certification fields"""
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
        return self.get_extra_dict().get('certification', {}).get(
            'certified_by')

    @property
    def certification_details(self):
        return self.get_extra_dict().get('certification', {}).get('details')

    @property
    def warning_markdown(self):
        return self.get_extra_dict().get('warning_markdown')


def clone_model(target, ignore=None, keep_relations=None, **kwargs: Any):
    """
    Clone a SQLAlchemy model. By default will only clone naive column attributes.
    To include relationship attributes, use `keep_relations`.
    """
    ignore = ignore or []
    table = target.__table__
    primary_keys = table.primary_key.columns.keys()
    data = {attr: getattr(target, attr) for attr in list(table.columns.keys
        ()) + (keep_relations or []) if attr not in primary_keys and attr
         not in ignore}
    data.update(kwargs)
    return target.__class__(**data)


class QueryStringExtended(NamedTuple):
    applied_template_filters: Optional[list[str]]
    applied_filter_columns: list[ColumnTyping]
    rejected_filter_columns: list[ColumnTyping]
    labels_expected: list[str]
    prequeries: list[str]
    sql: str


class SqlaQuery(NamedTuple):
    applied_template_filters: list[str]
    applied_filter_columns: list[ColumnTyping]
    rejected_filter_columns: list[ColumnTyping]
    cte: Optional[str]
    extra_cache_keys: list[Any]
    labels_expected: list[str]
    prequeries: list[str]
    sqla_query: Select


class ExploreMixin:
    """
    Allows any flask_appbuilder.Model (Query, Table, etc.)
    to be used to power a chart inside /explore
    """
    sqla_aggregations = {'COUNT_DISTINCT': lambda column_name: sa.func.
        COUNT(sa.distinct(column_name)), 'COUNT': sa.func.COUNT, 'SUM': sa.
        func.SUM, 'AVG': sa.func.AVG, 'MIN': sa.func.MIN, 'MAX': sa.func.MAX}
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

    def _process_sql_expression(self, expression, database_id, engine,
        schema, template_processor):
        if template_processor and expression:
            expression = template_processor.process_template(expression)
        if expression:
            expression = validate_adhoc_subquery(expression, database_id,
                engine, schema)
            try:
                expression = sanitize_clause(expression)
            except QueryClauseValidationException as ex:
                raise QueryObjectValidationError(ex.message) from ex
        return expression

    def make_sqla_column_compatible(self, sqla_col, label=None):
        """Takes a sqlalchemy column object and adds label info if supported by engine.
        :param sqla_col: sqlalchemy column instance
        :param label: alias/label that column is expected to have
        :return: either a sql alchemy column or label instance if supported by engine
        """
        label_expected = label or sqla_col.name
        db_engine_spec = self.db_engine_spec
        if db_engine_spec.get_allows_alias_in_select(self.database):
            label = db_engine_spec.make_label_compatible(label_expected)
            sqla_col = sqla_col.label(label)
        sqla_col.key = label_expected
        return sqla_col

    @staticmethod
    def _apply_cte(sql, cte):
        """
        Append a CTE before the SELECT statement if defined

        :param sql: SELECT statement
        :param cte: CTE statement
        :return:
        """
        if cte:
            sql = f'{cte}\n{sql}'
        return sql

    def get_query_str_extended(self, query_obj, mutate=True):
        sqlaq = self.get_sqla_query(**query_obj)
        sql = self.database.compile_sqla_query(sqlaq.sqla_query, catalog=
            self.catalog, schema=self.schema, is_virtual=bool(self.sql))
        sql = self._apply_cte(sql, sqlaq.cte)
        if mutate:
            sql = self.database.mutate_sql_based_on_config(sql)
        return QueryStringExtended(applied_template_filters=sqlaq.
            applied_template_filters, applied_filter_columns=sqlaq.
            applied_filter_columns, rejected_filter_columns=sqlaq.
            rejected_filter_columns, labels_expected=sqlaq.labels_expected,
            prequeries=sqlaq.prequeries, sql=sql)

    def _normalize_prequery_result_type(self, row, dimension, columns_by_name):
        """
        Convert a prequery result type to its equivalent Python type.

        Some databases like Druid will return timestamps as strings, but do not perform
        automatic casting when comparing these strings to a timestamp. For cases like
        this we convert the value via the appropriate SQL transform.

        :param row: A prequery record
        :param dimension: The dimension name
        :param columns_by_name: The mapping of columns by name
        :return: equivalent primitive python type
        """
        value = row[dimension]
        if isinstance(value, np.generic):
            value = value.item()
        column_ = columns_by_name[dimension]
        db_extra: dict[str, Any] = self.database.get_extra()
        if isinstance(column_, dict):
            if column_.get('type') and column_.get('is_temporal'
                ) and isinstance(value, str):
                sql = self.db_engine_spec.convert_dttm(column_.get('type'),
                    dateutil.parser.parse(value), db_extra=None)
                if sql:
                    value = self.db_engine_spec.get_text_clause(sql)
        elif column_.type and column_.is_temporal and isinstance(value, str):
            sql = self.db_engine_spec.convert_dttm(column_.type, dateutil.
                parser.parse(value), db_extra=db_extra)
            if sql:
                value = self.text(sql)
        return value

    def make_orderby_compatible(self, select_exprs, orderby_exprs):
        """
        If needed, make sure aliases for selected columns are not used in
        `ORDER BY`.

        In some databases (e.g. Presto), `ORDER BY` clause is not able to
        automatically pick the source column if a `SELECT` clause alias is named
        the same as a source column. In this case, we update the SELECT alias to
        another name to avoid the conflict.
        """
        if self.db_engine_spec.allows_alias_to_source_column:
            return

        def is_alias_used_in_orderby(col):
            if not isinstance(col, Label):
                return False
            regexp = re.compile(f'\\(.*\\b{re.escape(col.name)}\\b.*\\)',
                re.IGNORECASE)
            return any(regexp.search(str(x)) for x in orderby_exprs)
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
            """
            Some engines change the case or generate bespoke column names, either by
            default or due to lack of support for aliasing. This function ensures that
            the column names in the DataFrame correspond to what is expected by
            the viz components.
            Sometimes a query may also contain only order by columns that are not used
            as metrics or groupby columns, but need to present in the SQL `select`,
            filtering by `labels_expected` make sure we only return columns users want.
            :param df: Original DataFrame returned by the engine
            :return: Mutated DataFrame
            """
            labels_expected = query_str_ext.labels_expected
            if df is not None and not df.empty:
                if len(df.columns) < len(labels_expected):
                    raise QueryObjectValidationError(_(
                        'Db engine did not return all queried columns'))
                if len(df.columns) > len(labels_expected):
                    df = df.iloc[:, 0:len(labels_expected)]
                df.columns = labels_expected
            return df
        try:
            df = self.database.get_df(sql, self.catalog, self.schema,
                mutator=assign_column_label)
        except Exception as ex:
            df = pd.DataFrame()
            status = QueryStatus.FAILED
            logger.warning('Query %s on schema %s failed', sql, self.schema,
                exc_info=True)
            db_engine_spec = self.db_engine_spec
            errors = [dataclasses.asdict(error) for error in db_engine_spec
                .extract_errors(ex)]
            error_message = utils.error_msg_from_exception(ex)
        return QueryResult(applied_template_filters=query_str_ext.
            applied_template_filters, applied_filter_columns=query_str_ext.
            applied_filter_columns, rejected_filter_columns=query_str_ext.
            rejected_filter_columns, status=status, df=df, duration=
            datetime.now() - qry_start_dttm, query=sql, errors=errors,
            error_message=error_message)

    def get_rendered_sql(self, template_processor=None):
        """
        Render sql with template engine (Jinja).
        """
        if not self.sql:
            return ''
        sql = self.sql.strip('\t\r\n; ')
        if template_processor:
            try:
                sql = template_processor.process_template(sql)
            except TemplateError as ex:
                raise QueryObjectValidationError(_(
                    'Error while rendering virtual dataset query: %(msg)s',
                    msg=ex.message)) from ex
        script = SQLScript(sql, engine=self.db_engine_spec.engine)
        if len(script.statements) > 1:
            raise QueryObjectValidationError(_(
                'Virtual dataset query cannot consist of multiple statements'))
        if not sql:
            raise QueryObjectValidationError(_(
                'Virtual dataset query cannot be empty'))
        return sql

    def text(self, clause):
        return self.db_engine_spec.get_text_clause(clause)

    def get_from_clause(self, template_processor=None):
        """
        Return where to select the columns and metrics from. Either a physical table
        or a virtual table with it's own subquery. If the FROM is referencing a
        CTE, the CTE is returned as the second value in the return tuple.
        """
        from_sql = self.get_rendered_sql(template_processor) + '\n'
        parsed_script = SQLScript(from_sql, engine=self.db_engine_spec.engine)
        if parsed_script.has_mutation():
            raise QueryObjectValidationError(_(
                'Virtual dataset query must be read-only'))
        cte = self.db_engine_spec.get_cte_query(from_sql)
        from_clause = sa.table(self.db_engine_spec.cte_alias
            ) if cte else TextAsFrom(self.text(from_sql), []).alias(
            VIRTUAL_TABLE_ALIAS)
        return from_clause, cte

    def adhoc_metric_to_sqla(self, metric, columns_by_name,
        template_processor=None):
        """
        Turn an adhoc metric into a sqlalchemy column.

        :param dict metric: Adhoc metric definition
        :param dict columns_by_name: Columns for the current table
        :param template_processor: template_processor instance
        :returns: The metric defined as a sqlalchemy column
        :rtype: sqlalchemy.sql.column
        """
        expression_type = metric.get('expressionType')
        label = utils.get_metric_name(metric)
        if expression_type == utils.AdhocMetricExpressionType.SIMPLE:
            metric_column = metric.get('column') or {}
            column_name = cast(str, metric_column.get('column_name'))
            sqla_column = sa.column(column_name)
            sqla_metric = self.sqla_aggregations[metric['aggregate']](
                sqla_column)
        elif expression_type == utils.AdhocMetricExpressionType.SQL:
            expression = self._process_sql_expression(expression=metric[
                'sqlExpression'], database_id=self.database_id, engine=self
                .database.backend, schema=self.schema, template_processor=
                template_processor)
            sqla_metric = literal_column(expression)
        else:
            raise QueryObjectValidationError(
                'Adhoc metric expressionType is invalid')
        return self.make_sqla_column_compatible(sqla_metric, label)

    @property
    def template_params_dict(self):
        return {}

    @staticmethod
    def filter_values_handler(values, operator, target_generic_type,
        target_native_type=None, is_list_target=False, db_engine_spec=None,
        db_extra=None):
        if values is None:
            return None

        def handle_single_value(value):
            if operator == utils.FilterOperator.TEMPORAL_RANGE:
                return value
            if (isinstance(value, (float, int)) and target_generic_type ==
                utils.GenericDataType.TEMPORAL and target_native_type is not
                None and db_engine_spec is not None):
                value = db_engine_spec.convert_dttm(target_type=
                    target_native_type, dttm=datetime.utcfromtimestamp(
                    value / 1000), db_extra=db_extra)
                value = literal_column(value)
            if isinstance(value, str):
                value = value.strip('\t\n')
                if (target_generic_type == utils.GenericDataType.NUMERIC and
                    operator not in {utils.FilterOperator.ILIKE, utils.
                    FilterOperator.LIKE}):
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
        if is_list_target and not isinstance(values, (tuple, list)):
            values = [values]
        elif not is_list_target and isinstance(values, (tuple, list)):
            values = values[0] if values else None
        return values

    def get_query_str(self, query_obj):
        query_str_ext = self.get_query_str_extended(query_obj)
        all_queries = query_str_ext.prequeries + [query_str_ext.sql]
        return ';\n\n'.join(all_queries) + ';'

    def _get_series_orderby(self, series_limit_metric, metrics_by_name,
        columns_by_name, template_processor=None):
        if utils.is_adhoc_metric(series_limit_metric):
            assert isinstance(series_limit_metric, dict)
            ob = self.adhoc_metric_to_sqla(series_limit_metric, columns_by_name
                )
        elif isinstance(series_limit_metric, str
            ) and series_limit_metric in metrics_by_name:
            ob = metrics_by_name[series_limit_metric].get_sqla_col(
                template_processor=template_processor)
        else:
            raise QueryObjectValidationError(_(
                "Metric '%(metric)s' does not exist", metric=
                series_limit_metric))
        return ob

    def adhoc_column_to_sqla(self, col, force_type_check=False,
        template_processor=None):
        raise NotImplementedError()

    def _get_top_groups(self, df, dimensions, groupby_exprs, columns_by_name):
        groups = []
        for _unused, row in df.iterrows():
            group = []
            for dimension in dimensions:
                value = self._normalize_prequery_result_type(row, dimension,
                    columns_by_name)
                group.append(groupby_exprs[dimension] == value)
            groups.append(and_(*group))
        return or_(*groups)

    def dttm_sql_literal(self, dttm, col):
        """Convert datetime object to a SQL expression string"""
        sql = self.db_engine_spec.convert_dttm(col.type, dttm, db_extra=
            self.db_extra) if col.type else None
        if sql:
            return sql
        tf = col.python_date_format
        if not tf and self.db_extra:
            tf = self.db_extra.get('python_date_format_by_column_name', {}
                ).get(col.column_name)
        if tf:
            if tf in {'epoch_ms', 'epoch_s'}:
                seconds_since_epoch = int(dttm.timestamp())
                if tf == 'epoch_s':
                    return str(seconds_since_epoch)
                return str(seconds_since_epoch * 1000)
            return f"'{dttm.strftime(tf)}'"
        return f"'{dttm.strftime('%Y-%m-%d %H:%M:%S.%f')}'"

    def get_time_filter(self, time_col, start_dttm, end_dttm, time_grain=
        None, label='__time', template_processor=None):
        col = time_col.get_timestamp_expression(time_grain=time_grain,
            label=label, template_processor=template_processor
            ) if time_grain else self.convert_tbl_column_to_sqla_col(time_col,
            label=label, template_processor=template_processor)
        l = []
        if start_dttm:
            l.append(col >= self.db_engine_spec.get_text_clause(self.
                dttm_sql_literal(start_dttm, time_col)))
        if end_dttm:
            l.append(col < self.db_engine_spec.get_text_clause(self.
                dttm_sql_literal(end_dttm, time_col)))
        return and_(*l)

    def values_for_column(self, column_name, limit=10000,
        denormalize_column=False):
        db_dialect = self.database.get_dialect()
        column_name_ = self.database.db_engine_spec.denormalize_name(db_dialect
            , column_name) if denormalize_column else column_name
        cols = {col.column_name: col for col in self.columns}
        target_col = cols[column_name_]
        tp = self.get_template_processor()
        tbl, cte = self.get_from_clause(tp)
        qry = sa.select([target_col.get_sqla_col(template_processor=tp).
            label('column_values')]).select_from(tbl).distinct()
        if limit:
            qry = qry.limit(limit)
        if self.fetch_values_predicate:
            qry = qry.where(self.get_fetch_values_predicate(
                template_processor=tp))
        rls_filters = self.get_sqla_row_level_filters(template_processor=tp)
        qry = qry.where(and_(*rls_filters))
        with self.database.get_sqla_engine() as engine:
            sql = str(qry.compile(engine, compile_kwargs={'literal_binds': 
                True}))
            sql = self._apply_cte(sql, cte)
            sql = self.database.mutate_sql_based_on_config(sql)
            if engine.dialect.identifier_preparer._double_percents:
                sql = sql.replace('%%', '%')
            df = pd.read_sql_query(sql=sql, con=engine)
            df = df.replace({np.nan: None})
            return df['column_values'].to_list()

    def get_timestamp_expression(self, column, time_grain, label=None,
        template_processor=None):
        """
        Return a SQLAlchemy Core element representation of self to be used in a query.

        :param column: column object
        :param time_grain: Optional time grain, e.g. P1Y
        :param label: alias/label that column is expected to have
        :param template_processor: template processor
        :return: A TimeExpression object wrapped in a Label if supported by db
        """
        label = label or utils.DTTM_ALIAS
        column_spec = self.db_engine_spec.get_column_spec(column.get('type'))
        type_ = column_spec.sqla_type if column_spec else sa.DateTime
        col = sa.column(column.get('column_name'), type_=type_)
        if template_processor:
            expression = template_processor.process_template(column[
                'column_name'])
            col = sa.literal_column(expression, type_=type_)
        time_expr = self.db_engine_spec.get_timestamp_expr(col, None,
            time_grain)
        return self.make_sqla_column_compatible(time_expr, label)

    def convert_tbl_column_to_sqla_col(self, tbl_column, label=None,
        template_processor=None):
        label = label or tbl_column.column_name
        db_engine_spec = self.db_engine_spec
        column_spec = db_engine_spec.get_column_spec(self.type, db_extra=
            self.db_extra)
        type_ = column_spec.sqla_type if column_spec else None
        if (expression := tbl_column.expression):
            if template_processor:
                expression = template_processor.process_template(expression)
            col = literal_column(expression, type_=type_)
        else:
            col = sa.column(tbl_column.column_name, type_=type_)
        col = self.make_sqla_column_compatible(col, label)
        return col

    def get_sqla_query(self, apply_fetch_values_predicate=False, columns=
        None, extras=None, filter=None, from_dttm=None, granularity=None,
        groupby=None, inner_from_dttm=None, inner_to_dttm=None, is_rowcount
        =False, is_timeseries=True, metrics=None, orderby=None, order_desc=
        True, to_dttm=None, series_columns=None, series_limit=None,
        series_limit_metric=None, row_limit=None, row_offset=None,
        timeseries_limit=None, timeseries_limit_metric=None, time_shift=None):
        """Querying any sqla table from this common interface"""
        if granularity not in self.dttm_cols and granularity is not None:
            granularity = self.main_dttm_col
        extras = extras or {}
        time_grain = extras.get('time_grain_sqla')
        template_kwargs = {'columns': columns, 'from_dttm': from_dttm.
            isoformat() if from_dttm else None, 'groupby': groupby,
            'metrics': metrics, 'row_limit': row_limit, 'row_offset':
            row_offset, 'time_column': granularity, 'time_grain':
            time_grain, 'to_dttm': to_dttm.isoformat() if to_dttm else None,
            'table_columns': [col.column_name for col in self.columns],
            'filter': filter}
        columns = columns or []
        groupby = groupby or []
        rejected_adhoc_filters_columns: list[Union[str, ColumnTyping]] = []
        applied_adhoc_filters_columns: list[Union[str, ColumnTyping]] = []
        db_engine_spec = self.db_engine_spec
        series_column_labels = [db_engine_spec.make_label_compatible(column
            ) for column in utils.get_column_names(columns=series_columns or
            [])]
        if is_timeseries and timeseries_limit:
            series_limit = timeseries_limit
        series_limit_metric = series_limit_metric or timeseries_limit_metric
        template_kwargs.update(self.template_params_dict)
        extra_cache_keys: list[Any] = []
        template_kwargs['extra_cache_keys'] = extra_cache_keys
        removed_filters: list[str] = []
        applied_template_filters: list[str] = []
        template_kwargs['removed_filters'] = removed_filters
        template_kwargs['applied_filters'] = applied_template_filters
        template_processor = self.get_template_processor(**template_kwargs)
        prequeries: list[str] = []
        orderby = orderby or []
        need_groupby = bool(metrics is not None or groupby)
        metrics = metrics or []
        if granularity not in self.dttm_cols and granularity is not None:
            granularity = self.main_dttm_col
        columns_by_name: dict[str, 'TableColumn'] = {col.column_name: col for
            col in self.columns}
        metrics_by_name: dict[str, 'SqlMetric'] = {m.metric_name: m for m in
            self.metrics}
        if not granularity and is_timeseries:
            raise QueryObjectValidationError(_(
                'Datetime column not provided as part table configuration and is required by this type of chart'
                ))
        if not metrics and not columns and not groupby:
            raise QueryObjectValidationError(_('Empty query?'))
        metrics_exprs: list[ColumnElement] = []
        for metric in metrics:
            if utils.is_adhoc_metric(metric):
                assert isinstance(metric, dict)
                metrics_exprs.append(self.adhoc_metric_to_sqla(metric=
                    metric, columns_by_name=columns_by_name,
                    template_processor=template_processor))
            elif isinstance(metric, str) and metric in metrics_by_name:
                metrics_exprs.append(metrics_by_name[metric].get_sqla_col(
                    template_processor=template_processor))
            else:
                raise QueryObjectValidationError(_(
                    "Metric '%(metric)s' does not exist", metric=metric))
        if metrics_exprs:
            main_metric_expr = metrics_exprs[0]
        else:
            main_metric_expr, label = literal_column('COUNT(*)'), 'ccount'
            main_metric_expr = self.make_sqla_column_compatible(
                main_metric_expr, label)
        metrics_exprs_by_label = {m.key: m for m in metrics_exprs}
        metrics_exprs_by_expr = {str(m): m for m in metrics_exprs}
        orderby_exprs: list[ColumnElement] = []
        for orig_col, ascending in orderby:
            col: Union[AdhocMetric, ColumnElement] = orig_col
            if isinstance(col, dict):
                col = cast(AdhocMetric, col)
                if col.get('sqlExpression'):
                    col['sqlExpression'] = self._process_sql_expression(
                        expression=col['sqlExpression'], database_id=self.
                        database_id, engine=self.database.backend, schema=
                        self.schema, template_processor=template_processor)
                if utils.is_adhoc_metric(col):
                    col = self.adhoc_metric_to_sqla(col, columns_by_name)
                    col = metrics_exprs_by_expr.get(str(col), col)
                    need_groupby = True
            elif col in columns_by_name:
                col = self.convert_tbl_column_to_sqla_col(columns_by_name[
                    col], template_processor=template_processor)
            elif col in metrics_exprs_by_label:
                col = metrics_exprs_by_label[col]
                need_groupby = True
            elif col in metrics_by_name:
                col = metrics_by_name[col].get_sqla_col(template_processor=
                    template_processor)
                need_groupby = True
            if isinstance(col, ColumnElement):
                orderby_exprs.append(col)
            else:
                raise QueryObjectValidationError(_(
                    'Unknown column used in orderby: %(col)s', col=orig_col))
        select_exprs: list[Union[Column, Label]] = []
        groupby_all_columns = {}
        groupby_series_columns = {}
        columns = [col for col in columns if col != utils.DTTM_ALIAS]
        dttm_col = columns_by_name.get(granularity) if granularity else None
        if need_groupby:
            columns = groupby or columns
            for selected in columns:
                if isinstance(selected, str):
                    if selected == granularity:
                        table_col = columns_by_name[selected]
                        outer = table_col.get_timestamp_expression(time_grain
                            =time_grain, label=selected, template_processor
                            =template_processor)
                    elif selected in columns_by_name:
                        outer = self.convert_tbl_column_to_sqla_col(
                            columns_by_name[selected], template_processor=
                            template_processor)
                    else:
                        selected = validate_adhoc_subquery(selected, self.
                            database_id, self.database.backend, self.schema)
                        outer = literal_column(f'({selected})')
                        outer = self.make_sqla_column_compatible(outer,
                            selected)
                else:
                    outer = self.adhoc_column_to_sqla(col=selected,
                        template_processor=template_processor)
                groupby_all_columns[outer.name] = outer
                if (is_timeseries and not series_column_labels or outer.
                    name in series_column_labels):
                    groupby_series_columns[outer.name] = outer
                select_exprs.append(outer)
        elif columns:
            for selected in columns:
                if is_adhoc_column(selected):
                    _sql = selected['sqlExpression']
                    _column_label = selected['label']
                elif isinstance(selected, str):
                    _sql = selected
                    _column_label = selected
                selected = validate_adhoc_subquery(_sql, self.database_id,
                    self.database.backend, self.schema)
                select_exprs.append(self.convert_tbl_column_to_sqla_col(
                    columns_by_name[selected], template_processor=
                    template_processor, label=_column_label) if isinstance(
                    selected, str) and selected in columns_by_name else
                    self.make_sqla_column_compatible(literal_column(
                    selected), _column_label))
            metrics_exprs = []
        if granularity:
            if granularity not in columns_by_name or not dttm_col:
                raise QueryObjectValidationError(_(
                    'Time column "%(col)s" does not exist in dataset', col=
                    granularity))
            time_filters = []
            if is_timeseries:
                timestamp = dttm_col.get_timestamp_expression(time_grain=
                    time_grain, template_processor=template_processor)
                select_exprs.insert(0, timestamp)
                groupby_all_columns[timestamp.name] = timestamp
            if (self.always_filter_main_dttm and self.main_dttm_col in self
                .dttm_cols and self.main_dttm_col != dttm_col.column_name):
                time_filters.append(self.get_time_filter(time_col=
                    columns_by_name[self.main_dttm_col], start_dttm=
                    from_dttm, end_dttm=to_dttm, template_processor=
                    template_processor))
            time_filter_column = self.get_time_filter(time_col=dttm_col,
                start_dttm=from_dttm, end_dttm=to_dttm, template_processor=
                template_processor)
            time_filters.append(time_filter_column)
        select_exprs = remove_duplicates(select_exprs + metrics_exprs, key=
            lambda x: x.name)
        labels_expected = [c.key for c in select_exprs]
        if not db_engine_spec.allows_hidden_orderby_agg:
            select_exprs = remove_duplicates(select_exprs + orderby_exprs)
        qry = sa.select(select_exprs)
        tbl, cte = self.get_from_clause(template_processor)
        if groupby_all_columns:
            qry = qry.group_by(*groupby_all_columns.values())
        where_clause_and = []
        having_clause_and = []
        for flt in filter:
            if not all(flt.get(s) for s in ['col', 'op']):
                continue
            flt_col = flt['col']
            val = flt.get('val')
            flt_grain = flt.get('grain')
            op = flt['op'].upper()
            col_obj: Optional['TableColumn'] = None
            sqla_col: Optional[Column] = None
            if flt_col == utils.DTTM_ALIAS and is_timeseries and dttm_col:
                col_obj = dttm_col
            elif is_adhoc_column(flt_col):
                try:
                    sqla_col = self.adhoc_column_to_sqla(flt_col,
                        force_type_check=True)
                    applied_adhoc_filters_columns.append(flt_col)
                except ColumnNotFoundException:
                    rejected_adhoc_filters_columns.append(flt_col)
                    continue
            else:
                col_obj = columns_by_name.get(cast(str, flt_col))
            filter_grain = flt.get('grain')
            if get_column_name(flt_col) in removed_filters:
                continue
            if col_obj or sqla_col is not None:
                if sqla_col is not None:
                    pass
                elif col_obj and filter_grain:
                    sqla_col = col_obj.get_timestamp_expression(time_grain=
                        filter_grain, template_processor=template_processor)
                elif col_obj:
                    sqla_col = self.convert_tbl_column_to_sqla_col(tbl_column
                        =col_obj, template_processor=template_processor)
                col_type = col_obj.type if col_obj else None
                col_spec = db_engine_spec.get_column_spec(native_type=col_type)
                is_list_target = op in (utils.FilterOperator.IN.value,
                    utils.FilterOperator.NOT_IN.value)
                col_advanced_data_type = (col_obj.advanced_data_type if
                    col_obj else '')
                if col_spec and not col_advanced_data_type:
                    target_generic_type = col_spec.generic_type
                else:
                    target_generic_type = GenericDataType.STRING
                eq = self.filter_values_handler(values=val, operator=op,
                    target_generic_type=target_generic_type,
                    target_native_type=col_type, is_list_target=
                    is_list_target, db_engine_spec=db_engine_spec)
                if (col_advanced_data_type != '' and feature_flag_manager.
                    is_feature_enabled('ENABLE_ADVANCED_DATA_TYPES') and 
                    col_advanced_data_type in ADVANCED_DATA_TYPES):
                    values = eq if is_list_target else [eq]
                    bus_resp: AdvancedDataTypeResponse = ADVANCED_DATA_TYPES[
                        col_advanced_data_type].translate_type({'type':
                        col_advanced_data_type, 'values': values})
                    if bus_resp['error_message']:
                        raise AdvancedDataTypeResponseError(_(bus_resp[
                            'error_message']))
                    where_clause_and.append(ADVANCED_DATA_TYPES[
                        col_advanced_data_type].translate_filter(sqla_col,
                        op, bus_resp['values']))
                elif is_list_target:
                    assert isinstance(eq, (tuple, list))
                    if len(eq) == 0:
                        raise QueryObjectValidationError(_(
                            'Filter value list cannot be empty'))
                    if len(eq) > len((eq_without_none := [x for x in eq if 
                        x is not None])):
                        is_null_cond = sqla_col.is_(None)
                        if eq:
                            cond = or_(is_null_cond, sqla_col.in_(
                                eq_without_none))
                        else:
                            cond = is_null_cond
                    else:
                        cond = sqla_col.in_(eq)
                    if op == utils.FilterOperator.NOT_IN.value:
                        cond = ~cond
                    where_clause_and.append(cond)
                elif op == utils.FilterOperator.IS_NULL.value:
                    where_clause_and.append(sqla_col.is_(None))
                elif op == utils.FilterOperator.IS_NOT_NULL.value:
                    where_clause_and.append(sqla_col.isnot(None))
                elif op == utils.FilterOperator.IS_TRUE.value:
                    where_clause_and.append(sqla_col.is_(True))
                elif op == utils.FilterOperator.IS_FALSE.value:
                    where_clause_and.append(sqla_col.is_(False))
                else:
                    if op not in {utils.FilterOperator.EQUALS.value, utils.
                        FilterOperator.NOT_EQUALS.value} and eq is None:
                        raise QueryObjectValidationError(_(
                            'Must specify a value for filters with comparison operators'
                            ))
                    if op == utils.FilterOperator.EQUALS.value:
                        where_clause_and.append(sqla_col == eq)
                    elif op == utils.FilterOperator.NOT_EQUALS.value:
                        where_clause_and.append(sqla_col != eq)
                    elif op == utils.FilterOperator.GREATER_THAN.value:
                        where_clause_and.append(sqla_col > eq)
                    elif op == utils.FilterOperator.LESS_THAN.value:
                        where_clause_and.append(sqla_col < eq)
                    elif op == utils.FilterOperator.GREATER_THAN_OR_EQUALS.value:
                        where_clause_and.append(sqla_col >= eq)
                    elif op == utils.FilterOperator.LESS_THAN_OR_EQUALS.value:
                        where_clause_and.append(sqla_col <= eq)
                    elif op in {utils.FilterOperator.ILIKE.value, utils.
                        FilterOperator.LIKE.value}:
                        if target_generic_type != GenericDataType.STRING:
                            sqla_col = sa.cast(sqla_col, sa.String)
                        if op == utils.FilterOperator.LIKE.value:
                            where_clause_and.append(sqla_col.like(eq))
                        else:
                            where_clause_and.append(sqla_col.ilike(eq))
                    elif op in {utils.FilterOperator.NOT_LIKE.value}:
                        if target_generic_type != GenericDataType.STRING:
                            sqla_col = sa.cast(sqla_col, sa.String)
                        where_clause_and.append(sqla_col.not_like(eq))
                    elif op == utils.FilterOperator.TEMPORAL_RANGE.value and isinstance(
                        eq, str) and col_obj is not None:
                        _since, _until = get_since_until_from_time_range(
                            time_range=eq, time_shift=time_shift, extras=extras
                            )
                        where_clause_and.append(self.get_time_filter(
                            time_col=col_obj, start_dttm=_since, end_dttm=
                            _until, time_grain=flt_grain, label=sqla_col.
                            key, template_processor=template_processor))
                    else:
                        raise QueryObjectValidationError(_(
                            'Invalid filter operation type: %(op)s', op=op))
        where_clause_and += self.get_sqla_row_level_filters(template_processor)
        if extras:
            where = extras.get('where')
            if where:
                try:
                    where = template_processor.process_template(f'({where})')
                except TemplateError as ex:
                    raise QueryObjectValidationError(_(
                        'Error in jinja expression in WHERE clause: %(msg)s',
                        msg=ex.message)) from ex
                where = self._process_sql_expression(expression=where,
                    database_id=self.database_id, engine=self.database.
                    backend, schema=self.schema, template_processor=
                    template_processor)
                where_clause_and += [self.text(where)]
            having = extras.get('having')
            if having:
                try:
                    having = template_processor.process_template(f'({having})')
                except TemplateError as ex:
                    raise QueryObjectValidationError(_(
                        'Error in jinja expression in HAVING clause: %(msg)s',
                        msg=ex.message)) from ex
                having = self._process_sql_expression(expression=having,
                    database_id=self.database_id, engine=self.database.
                    backend, schema=self.schema, template_processor=
                    template_processor)
                having_clause_and += [self.text(having)]
        if apply_fetch_values_predicate and self.fetch_values_predicate:
            qry = qry.where(self.get_fetch_values_predicate(
                template_processor=template_processor))
        if granularity:
            qry = qry.where(and_(*(time_filters + where_clause_and)))
        else:
            qry = qry.where(and_(*where_clause_and))
        qry = qry.having(and_(*having_clause_and))
        self.make_orderby_compatible(select_exprs, orderby_exprs)
        for col, (orig_col, ascending) in zip(orderby_exprs, orderby,
            strict=False):
            if not db_engine_spec.allows_alias_in_orderby and isinstance(col,
                Label):
                col = col.element
            if db_engine_spec.get_allows_alias_in_select(self.database
                ) and db_engine_spec.allows_hidden_cc_in_orderby and col.name in [
                select_col.name for select_col in select_exprs]:
                with self.database.get_sqla_engine() as engine:
                    quote = engine.dialect.identifier_preparer.quote
                    col = literal_column(quote(col.name))
            direction = sa.asc if ascending else sa.desc
            qry = qry.order_by(direction(col))
        if row_limit:
            qry = qry.limit(row_limit)
        if row_offset:
            qry = qry.offset(row_offset)
        if series_limit and groupby_series_columns:
            if (db_engine_spec.allows_joins and db_engine_spec.
                allows_subqueries):
                inner_main_metric_expr = self.make_sqla_column_compatible(
                    main_metric_expr, 'mme_inner__')
                inner_groupby_exprs = []
                inner_select_exprs = []
                for gby_name, gby_obj in groupby_series_columns.items():
                    inner = self.make_sqla_column_compatible(gby_obj, 
                        gby_name + '__')
                    inner_groupby_exprs.append(inner)
                    inner_select_exprs.append(inner)
                inner_select_exprs += [inner_main_metric_expr]
                subq = sa.select(inner_select_exprs).select_from(tbl)
                inner_time_filter = []
                if dttm_col and not db_engine_spec.time_groupby_inline:
                    inner_time_filter = [self.get_time_filter(time_col=
                        dttm_col, start_dttm=inner_from_dttm or from_dttm,
                        end_dttm=inner_to_dttm or to_dttm,
                        template_processor=template_processor)]
                subq = subq.where(and_(*(where_clause_and + inner_time_filter))
                    )
                subq = subq.group_by(*inner_groupby_exprs)
                ob = inner_main_metric_expr
                if series_limit_metric:
                    ob = self._get_series_orderby(series_limit_metric=
                        series_limit_metric, metrics_by_name=
                        metrics_by_name, columns_by_name=columns_by_name,
                        template_processor=template_processor)
                direction = sa.desc if order_desc else sa.asc
                subq = subq.order_by(direction(ob))
                subq = subq.limit(series_limit)
                on_clause = []
                for gby_name, gby_obj in groupby_series_columns.items():
                    col_name = db_engine_spec.make_label_compatible(
                        gby_name + '__')
                    on_clause.append(gby_obj == sa.column(col_name))
                tbl = tbl.join(subq.alias(SERIES_LIMIT_SUBQ_ALIAS), and_(*
                    on_clause))
            else:
                if series_limit_metric:
                    orderby = [(self._get_series_orderby(
                        series_limit_metric=series_limit_metric,
                        metrics_by_name=metrics_by_name, columns_by_name=
                        columns_by_name, template_processor=
                        template_processor), not order_desc)]
                prequery_obj = {'is_timeseries': False, 'row_limit':
                    series_limit, 'metrics': metrics, 'granularity':
                    granularity, 'groupby': groupby, 'from_dttm': 
                    inner_from_dttm or from_dttm, 'to_dttm': inner_to_dttm or
                    to_dttm, 'filter': filter, 'orderby': orderby, 'extras':
                    extras, 'columns': get_non_base_axis_columns(columns),
                    'order_desc': True}
                result = self.query(prequery_obj)
                prequeries.append(result.query)
                dimensions = [c for c in result.df.columns if c not in
                    metrics and c in groupby_series_columns]
                top_groups = self._get_top_groups(result.df, dimensions,
                    groupby_series_columns, columns_by_name)
                qry = qry.where(top_groups)
        qry = qry.select_from(tbl)
        if is_rowcount:
            if not db_engine_spec.allows_subqueries:
                raise QueryObjectValidationError(_(
                    'Database does not support subqueries'))
            label = 'rowcount'
            col = self.make_sqla_column_compatible(literal_column(
                'COUNT(*)'), label)
            qry = sa.select([col]).select_from(qry.alias('rowcount_qry'))
            labels_expected = [label]
        filter_columns = [flt.get('col') for flt in filter] if filter else []
        rejected_filter_columns = [col for col in filter_columns if col and
            not is_adhoc_column(col) and col not in self.column_names and 
            col not in applied_template_filters
            ] + rejected_adhoc_filters_columns
        applied_filter_columns = [col for col in filter_columns if col and 
            not is_adhoc_column(col) and (col in self.column_names or col in
            applied_template_filters)] + applied_adhoc_filters_columns
        return SqlaQuery(applied_template_filters=applied_template_filters,
            cte=cte, applied_filter_columns=applied_filter_columns,
            rejected_filter_columns=rejected_filter_columns,
            extra_cache_keys=extra_cache_keys, labels_expected=
            labels_expected, sqla_query=qry, prequeries=prequeries)
