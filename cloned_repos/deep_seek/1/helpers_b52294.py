"""a collection of model-related helper classes and functions"""
import builtins
import dataclasses
import logging
import re
import uuid
from collections.abc import Hashable
from datetime import datetime, timedelta
from typing import Any, cast, NamedTuple, Optional, TYPE_CHECKING, Union, Dict, List, Set, Tuple, Callable, TypeVar, Type, Sequence, Mapping, Iterable
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

T = TypeVar('T')
ColumnType = Union[str, Dict[str, Any]]
FilterType = Dict[str, Any]
OrderByType = Tuple[ColumnType, bool]
QueryObjectType = Dict[str, Any]

def validate_adhoc_subquery(sql: str, database_id: int, engine: Any, default_schema: str) -> str:
    """
    Check if adhoc SQL contains sub-queries or nested sub-queries with table.
    """
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

def json_to_dict(json_str: Optional[str]) -> Dict[str, Any]:
    if json_str:
        val = re.sub(',[ \t\r\n]+}', '}', json_str)
        val = re.sub(',[ \t\r\n]+\\]', ']', val)
        return json.loads(val)
    return {}

def convert_uuids(obj: Any) -> Any:
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
    uuid = sa.Column(UUIDType(binary=True), primary_key=False, unique=True, default=uuid.uuid4)

    @property
    def short_uuid(self) -> str:
        return str(self.uuid)[:8]

class ImportExportMixin(UUIDMixin):
    export_parent: Optional[str] = None
    export_children: List[str] = []
    export_fields: List[str] = []
    extra_import_fields: List[str] = []

    @classmethod
    def _unique_constraints(cls) -> List[Set[str]]:
        """Get all (single column and multi column) unique constraints"""
        unique = [{c.name for c in u.columns} for u in cls.__table_args__ if isinstance(u, UniqueConstraint)]
        unique.extend(({c.name} for c in cls.__table__.columns if c.unique))
        return unique

    @classmethod
    def parent_foreign_key_mappings(cls) -> Dict[str, str]:
        """Get a mapping of foreign name to the local name of foreign keys"""
        parent_rel = cls.__mapper__.relationships.get(cls.export_parent)
        if parent_rel:
            return {local.name: remote.name for local, remote in parent_rel.local_remote_pairs}
        return {}

    @classmethod
    def export_schema(cls, recursive: bool = True, include_parent_ref: bool = False) -> Dict[str, Any]:
        """Export schema as a dictionary"""
        parent_excludes = set()
        if not include_parent_ref:
            parent_ref = cls.__mapper__.relationships.get(cls.export_parent)
            if parent_ref:
                parent_excludes = {column.name for column in parent_ref.local_columns}

        def formatter(column: Column) -> str:
            return f'{str(column.type)} Default ({column.default.arg})' if column.default else str(column.type)
        schema = {column.name: formatter(column) for column in cls.__table__.columns if column.name in cls.export_fields and column.name not in parent_excludes}
        if recursive:
            for column in cls.export_children:
                child_class = cls.__mapper__.relationships[column].argument.class_
                schema[column] = [child_class.export_schema(recursive=recursive, include_parent_ref=include_parent_ref)]
        return schema

    @classmethod
    def import_from_dict(cls, dict_rep: Dict[str, Any], parent: Optional[Any] = None, recursive: bool = True, sync: Optional[List[str]] = None, allow_reparenting: bool = False) -> Any:
        """Import obj from a dictionary"""
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

    def export_to_dict(self, recursive: bool = True, include_parent_ref: bool = False, include_defaults: bool = False, export_uuids: bool = False) -> Dict[str, Any]:
        """Export obj to dictionary"""
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

    def override(self, obj: Any) -> None:
        """Overrides the plain fields of the dashboard."""
        for field in obj.__class__.export_fields:
            setattr(self, field, getattr(obj, field))

    def copy(self) -> Any:
        """Creates a copy of the dashboard without relationships."""
        new_obj = self.__class__()
        new_obj.override(self)
        return new_obj

    def alter_params(self, **kwargs: Any) -> None:
        params = self.params_dict
        params.update(kwargs)
        self.params = json.dumps(params)

    def remove_params(self, param_to_remove: str) -> None:
        params = self.params_dict
        params.pop(param_to_remove, None)
        self.params = json.dumps(params)

    def reset_ownership(self) -> None:
        """object will belong to the user the current user"""
        self.created_by = None
        self.changed_by = None
        self.owners = []
        if g and hasattr(g, 'user'):
            self.owners = [g.user]

    @property
    def params_dict(self) -> Dict[str, Any]:
        return json_to_dict(self.params)

    @property
    def template_params_dict(self) -> Dict[str, Any]:
        return json_to_dict(self.template_params)

def _user(user: Optional[User]) -> str:
    if not user:
        return ''
    return escape(user)

class AuditMixinNullable(AuditMixin):
    """Altering the AuditMixin to use nullable fields"""
    created_on = sa.Column(sa.DateTime, default=datetime.now, nullable=True)
    changed_on = sa.Column(sa.DateTime, default=datetime.now, onupdate=datetime.now, nullable=True)

    @declared_attr
    def created_by_fk(self) -> Column:
        return sa.Column(sa.Integer, sa.ForeignKey('ab_user.id'), default=get_user_id, nullable=True)

    @declared_attr
    def changed_by_fk(self) -> Column:
        return sa.Column(sa.Integer, sa.ForeignKey('ab_user.id'), default=get_user_id, onupdate=get_user_id, nullable=True)

    @property
    def created_by_name(self) -> str:
        if self.created_by:
            return escape(f'{self.created_by}')
        return ''

    @property
    def changed_by_name(self) -> str:
        if self.changed_by:
            return escape(f'{self.changed_by}')
        return ''

    @renders('created_by')
    def creator(self) -> str:
        return _user(self.created_by)

    @property
    def changed_by_(self) -> str:
        return _user(self.changed_by)

    @renders('changed_on')
    def changed_on_(self) -> Markup:
        return Markup(f'<span class="no-wrap">{self.changed_on}</span>')

    @renders('changed_on')
    def changed_on_delta_humanized(self) -> str:
        return self.changed_on_humanized

    @renders('changed_on')
    def changed_on_dttm(self) -> float:
        return datetime_to_epoch(self.changed_on)

    @renders('created_on')
    def created_on_delta_humanized(self) -> str:
        return self.created_on_humanized

    @renders('changed_on')
    def changed_on_utc(self) -> str:
        return self.changed_on.astimezone(pytz.utc).strftime('%Y-%m-%dT%H:%M:%S.%f%z')

    @property
    def changed_on_humanized(self) -> str:
        return humanize.naturaltime(datetime.now() - self.changed_on)

    @property
    def created_on_humanized(self) -> str:
        return humanize.naturaltime(datetime.now() - self.created_on)

    @renders('changed_on')
    def modified(self) -> Markup:
        return Markup(f'<span class="no-wrap">{self.changed_on_humanized}</span>')

class QueryResult:
    """Object returned by the query interface"""

    def __init__(
        self,
        df: pd.DataFrame,
        query: str,
        duration: timedelta,
        applied_template_filters: Optional[List[str]] = None,
        applied_filter_columns: Optional[List[str]] = None,
        rejected_filter_columns: Optional[List[str]] = None,
        status: QueryStatus = QueryStatus.SUCCESS,
        error_message: Optional[str] = None,
        errors: Optional[List[Dict[str, Any]]] = None,
        from_dttm: Optional[datetime] = None,
        to_dttm: Optional[datetime] = None
    ):
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
    def extra(self) -> Dict[str, Any]:
        try:
            return json.loads(self.extra_json or '{}') or {}
        except (TypeError, json.JSONDecodeError) as exc:
            logger.error('Unable to load an extra json: %r. Leaving empty.', exc, exc_info=True)
            return {}

    @extra.setter
    def extra(self, extras: Dict[str, Any]) -> None:
        self.extra_json = json.dumps(extras)

    def set_extra_json_key(self, key: str, value: Any) ->