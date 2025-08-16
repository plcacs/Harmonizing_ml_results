from __future__ import annotations
import re
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache, partial
from typing import Any, Callable, cast, Optional, TYPE_CHECKING, TypedDict, Union
import dateutil
from flask import current_app, g, has_request_context, request
from flask_babel import gettext as _
from jinja2 import DebugUndefined, Environment, nodes
from jinja2.nodes import Call, Node
from jinja2.sandbox import SandboxedEnvironment
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.sql.expression import bindparam
from sqlalchemy.types import String
from superset.commands.dataset.exceptions import DatasetNotFoundError
from superset.common.utils.time_range_utils import get_since_until_from_time_range
from superset.constants import LRU_CACHE_MAX_SIZE, NO_TIME_RANGE
from superset.exceptions import SupersetTemplateException
from superset.extensions import feature_flag_manager
from superset.sql_parse import Table
from superset.utils import json
from superset.utils.core import AdhocFilterClause, convert_legacy_filters_into_adhoc, FilterOperator, get_user_email, get_user_id, get_username, merge_extra_filters

if TYPE_CHECKING:
    from superset.connectors.sqla.models import SqlaTable
    from superset.models.core import Database
    from superset.models.sql_lab import Query

NONE_TYPE: str = type(None).__name__
ALLOWED_TYPES: tuple[str] = (NONE_TYPE, 'bool', 'str', 'unicode', 'int', 'long', 'float', 'list', 'dict', 'tuple', 'set', 'TimeFilter')
COLLECTION_TYPES: tuple[str] = ('list', 'dict', 'tuple', 'set')

@lru_cache(maxsize=LRU_CACHE_MAX_SIZE)
def context_addons() -> dict:
    return current_app.config.get('JINJA_CONTEXT_ADDONS', {})

class Filter(TypedDict):
    pass

@dataclass
class TimeFilter:
    """
    Container for temporal filter.
    """

class ExtraCache:
    """
    Dummy class that exposes a method used to store additional values used in
    calculation of query object cache keys.
    """
    regex: re.Pattern = re.compile('(\\{\\{|\\{%)[^{}]*?(current_user_id\\([^()]*\\)|current_username\\([^()]*\\)|current_user_email\\([^()]*\\)|cache_key_wrapper\\([^()]*\\)|url_param\\([^()]*\\))[^{}]*?(\\}\\}|\\%\\})')

    def __init__(self, extra_cache_keys: Optional[list] = None, applied_filters: Optional[list] = None, removed_filters: Optional[list] = None, database: Optional[Database] = None, dialect: Optional[Dialect] = None, table: Optional[SqlaTable] = None) -> None:
        self.extra_cache_keys = extra_cache_keys
        self.applied_filters = applied_filters if applied_filters is not None else []
        self.removed_filters = removed_filters if removed_filters is not None else []
        self.database = database
        self.dialect = dialect
        self.table = table

    def current_user_id(self, add_to_cache_keys: bool = True) -> Optional[int]:
        if (user_id := get_user_id()):
            if add_to_cache_keys:
                self.cache_key_wrapper(user_id)
            return user_id
        return None

    def current_username(self, add_to_cache_keys: bool = True) -> Optional[str]:
        if (username := get_username()):
            if add_to_cache_keys:
                self.cache_key_wrapper(username)
            return username
        return None

    def current_user_email(self, add_to_cache_keys: bool = True) -> Optional[str]:
        if (email_address := get_user_email()):
            if add_to_cache_keys:
                self.cache_key_wrapper(email_address)
            return email_address
        return None

    def cache_key_wrapper(self, key: Any) -> Any:
        if self.extra_cache_keys is not None:
            self.extra_cache_keys.append(key)
        return key

    def url_param(self, param: str, default: Any = None, add_to_cache_keys: bool = True, escape_result: bool = True) -> Any:
        from superset.views.utils import get_form_data
        if has_request_context() and request.args.get(param):
            return request.args.get(param, default)
        form_data, _ = get_form_data()
        url_params = form_data.get('url_params') or {}
        result = url_params.get(param, default)
        if result and escape_result and self.dialect:
            result = String().literal_processor(dialect=self.dialect)(value=result)[1:-1]
        if add_to_cache_keys:
            self.cache_key_wrapper(result)
        return result

    def filter_values(self, column: str, default: Any = None, remove_filter: bool = False) -> list:
        return_val: list = []
        filters = self.get_filters(column, remove_filter)
        for flt in filters:
            val = flt.get('val')
            if isinstance(val, list):
                return_val.extend(val)
            elif val:
                return_val.append(val)
        if not return_val and default:
            return_val = [default]
        return return_val

    def get_filters(self, column: str, remove_filter: bool = False) -> list:
        from superset.views.utils import get_form_data
        form_data, _ = get_form_data()
        convert_legacy_filters_into_adhoc(form_data)
        merge_extra_filters(form_data)
        filters = []
        for flt in form_data.get('adhoc_filters', []):
            val = flt.get('comparator')
            op = flt['operator'].upper() if flt.get('operator') else None
            if flt.get('expressionType') == 'SIMPLE' and flt.get('clause') == 'WHERE' and (flt.get('subject') == column) and val:
                if remove_filter:
                    if column not in self.removed_filters:
                        self.removed_filters.append(column)
                if column not in self.applied_filters:
                    self.applied_filters.append(column)
                if op in (FilterOperator.IN.value, FilterOperator.NOT_IN.value) and (not isinstance(val, list)):
                    val = [val]
                filters.append({'op': op, 'col': column, 'val': val})
        return filters

    def get_time_filter(self, column: Optional[str] = None, default: Any = None, target_type: Optional[str] = None, strftime: Optional[str] = None, remove_filter: bool = False) -> TimeFilter:
        from superset.views.utils import get_form_data
        form_data, _ = get_form_data()
        convert_legacy_filters_into_adhoc(form_data)
        merge_extra_filters(form_data)
        time_range = form_data.get('time_range')
        if column:
            flt = next((flt for flt in form_data.get('adhoc_filters', []) if flt['operator'] == FilterOperator.TEMPORAL_RANGE and flt['subject'] == column), None)
            if flt:
                if remove_filter:
                    if column not in self.removed_filters:
                        self.removed_filters.append(column)
                if column not in self.applied_filters:
                    self.applied_filters.append(column)
                time_range = cast(str, flt['comparator'])
                if not target_type and self.table:
                    target_type = self.table.columns_types.get(column)
        time_range = time_range or NO_TIME_RANGE
        if time_range == NO_TIME_RANGE and default:
            time_range = default
        from_expr, to_expr = get_since_until_from_time_range(time_range)

        def _format_dttm(dttm: datetime) -> Optional[str]:
            if strftime and dttm:
                return dttm.strftime(strftime)
            return self.database.db_engine_spec.convert_dttm(target_type or '', dttm) if self.database and dttm else None
        return TimeFilter(from_expr=_format_dttm(from_expr), to_expr=_format_dttm(to_expr), time_range=time_range)

def safe_proxy(func: Callable, *args: Any, **kwargs: Any) -> Any:
    return_value = func(*args, **kwargs)
    value_type = type(return_value).__name__
    if value_type not in ALLOWED_TYPES:
        raise SupersetTemplateException(_('Unsafe return type for function %(func)s: %(value_type)s', func=func.__name__, value_type=value_type))
    if value_type in COLLECTION_TYPES:
        try:
            return_value = json.loads(json.dumps(return_value))
        except TypeError as ex:
            raise SupersetTemplateException(_('Unsupported return value for method %(name)s', name=func.__name__)) from ex
    return return_value

def validate_context_types(context: dict) -> dict:
    for key in context:
        arg_type = type(context[key]).__name__
        if arg_type not in ALLOWED_TYPES and key not in context_addons():
            if arg_type == 'partial' and context[key].func.__name__ == 'safe_proxy':
                continue
            raise SupersetTemplateException(_('Unsafe template value for key %(key)s: %(value_type)s', key=key, value_type=arg_type))
        if arg_type in COLLECTION_TYPES:
            try:
                context[key] = json.loads(json.dumps(context[key]))
            except TypeError as ex:
                raise SupersetTemplateException(_('Unsupported template value for key %(key)s', key=key)) from ex
    return context

def validate_template_context(engine: str, context: dict) -> dict:
    if engine and engine in context:
        engine_context = validate_context_types(context.pop(engine))
        valid_context = validate_context_types(context)
        valid_context[engine] = engine_context
        return valid_context
    return validate_context_types(context)

class WhereInMacro:

    def __init__(self, dialect: Dialect) -> None:
        self.dialect = dialect

    def __call__(self, values: list, mark: Optional[Any] = None) -> str:
        binds = [bindparam(f'value_{i}', value) for i, value in enumerate(values)]
        string_representations = [str(bind.compile(dialect=self.dialect, compile_kwargs={'literal_binds': True})) for bind in binds]
        joined_values = ', '.join(string_representations)
        result = f'({joined_values})'
        if mark:
            result += '\n-- WARNING: the `mark` parameter was removed from the `where_in` macro for security reasons\n'
        return result

class BaseTemplateProcessor:
    """
    Base class for database-specific jinja context
    """
    engine: Optional[str] = None

    def __init__(self, database: Database, query: Optional[Query] = None, table: Optional[SqlaTable] = None, extra_cache_keys: Optional[list] = None, removed_filters: Optional[list] = None, applied_filters: Optional[list] = None, **kwargs: Any) -> None:
        self._database = database
        self._query = query
        self._schema = None
        if query and query.schema:
            self._schema = query.schema
        elif table:
            self._schema = table.schema
        self._table = table
        self._extra_cache_keys = extra_cache_keys
        self._applied_filters = applied_filters
        self._removed_filters = removed_filters
        self._context = {}
        self.env = SandboxedEnvironment(undefined=DebugUndefined)
        self.set_context(**kwargs)
        self.env.filters['where_in'] = WhereInMacro(database.get_dialect())

    def set_context(self, **kwargs: Any) -> None:
        self._context.update(kwargs)
        self._context.update(context_addons())

    def process_template(self, sql: str, **kwargs: Any) -> str:
        """Processes a sql template

        >>> sql = "SELECT '{{ datetime(2017, 1, 1).isoformat() }}'"
        >>> process_template(sql)
        "SELECT '2017-01-01T00:00:00'"
        """
        template = self.env.from_string(sql)
        kwargs.update(self._context)
        context = validate_template_context(self.engine, kwargs)
        return template.render(context)

class JinjaTemplateProcessor(BaseTemplateProcessor):

    def _parse_datetime(self, dttm: str) -> Optional[datetime]:
        """
        Try to parse a datetime and default to None in the worst case.

        Since this may have been rendered by different engines, the datetime may
        vary slightly in format. We try to make it consistent, and if all else
        fails, just return None.
        """
        try:
            return dateutil.parser.parse(dttm)
        except dateutil.parser.ParserError:
            return None

    def set_context(self, **kwargs: Any) -> None:
        super().set_context(**kwargs)
        extra_cache = ExtraCache(extra_cache_keys=self._extra_cache_keys, applied_filters=self._applied_filters, removed_filters=self._removed_filters, database=self._database, dialect=self._database.get_dialect(), table=self._table)
        from_dttm = self._parse_datetime(dttm) if (dttm := self._context.get('from_dttm')) else None
        to_dttm = self._parse_datetime(dttm) if (dttm := self._context.get('to_dttm')) else None
        dataset_macro_with_context = partial(dataset_macro, from_dttm=from_dttm, to_dttm=to_dttm)
        self._context.update({'url_param': partial(safe_proxy, extra_cache.url_param), 'current_user_id': partial(safe_proxy, extra_cache.current_user_id), 'current_username': partial(safe_proxy, extra_cache.current_username), 'current_user_email': partial(safe_proxy, extra_cache.current_user_email), 'cache_key_wrapper': partial(safe_proxy, extra_cache.cache_key_wrapper), 'filter_values': partial(safe_proxy, extra_cache.filter_values), 'get_filters': partial(safe_proxy, extra_cache.get_filters), 'dataset': partial(safe_proxy, dataset_macro_with_context), 'metric': partial(safe_proxy, metric_macro), 'get_time_filter': partial(safe_proxy, extra_cache.get_time_filter)}

class NoOpTemplateProcessor(BaseTemplateProcessor):

    def process_template(self, sql: str, **kwargs: Any) -> str:
        """
        Makes processing a template a noop
        """
        return str(sql)

class PrestoTemplateProcessor(JinjaTemplateProcessor):
    """Presto Jinja context

    The methods described here are namespaced under ``presto`` in the
    jinja context as in ``SELECT '{{ presto.some_macro_call() }}'``
    """
    engine: str = 'presto'

    def set_context(self, **kwargs: Any) -> None:
        super().set_context(**kwargs)
        self._context[self.engine] = {'first_latest_partition': partial(safe_proxy, self.first_latest_partition), 'latest_partitions': partial(safe_proxy, self.latest_partitions), 'latest_sub_partition': partial(safe_proxy, self.latest_sub_partition), 'latest_partition': partial(safe_proxy, self.latest_partition)}

    @staticmethod
    def _schema_table(table_name: str, schema: Optional[str]) -> tuple[str, Optional[str]]:
        if '.' in table_name:
            schema, table_name = table_name.split('.')
        return (table_name, schema)

    def first_latest_partition(self, table_name: str) -> Optional[str]:
        """
        Gets the first value in the array of all latest partitions

        :param table_name: table name in the format `schema.table`
        :return: the first (or only) value in the latest partition array
        :raises IndexError: If no partition exists
        """
        latest_partitions = self.latest_partitions(table_name)
        return latest_partitions[0] if latest_partitions else None

    def latest_partitions(self, table_name: str) -> list:
        """
        Gets the array of all latest partitions

        :param table_name: table name in the format `schema.table`
        :return: the latest partition array
        """
        from superset.db_engine_specs.presto import PrestoEngineSpec
        table_name, schema = self._schema_table(table_name, self._schema)
        return cast(PrestoEngineSpec, self._database.db_engine_spec).latest_partition(database=self._database, table=Table(table_name, schema))[1]

    def latest_sub_partition(self, table_name: str, **kwargs: Any) -> Optional[str]:
        table_name, schema = self._schema_table(table_name, self._schema)
        from superset.db_engine_specs.presto import PrestoEngineSpec
        return cast(PrestoEngineSpec, self._database.db_engine_spec).latest_sub_partition(database=self._database, table=Table(table_name, schema), **kwargs)
    latest_partition = first_latest_partition

class HiveTemplateProcessor(PrestoTemplateProcessor):
    engine: str = 'hive'

class SparkTemplateProcessor(HiveTemplateProcessor):
    engine: str = 'spark'

    def process_template(self, sql: str, **kwargs: Any) -> str:
        template = self.env.from_string(sql)
        kwargs.update(self._context)
        context = validate_template_context(self.engine, kwargs)
        context['hive'] = context['spark']
        return template.render(context)

class TrinoTemplateProcessor(PrestoTemplateProcessor):
    engine: str = 'trino'

    def process_template(self, sql: str, **kwargs: Any) -> str:
        template = self.env.from_string(sql)
        kwargs.update(self._context)
        context = validate_template_context(self.engine, kwargs)
        context['presto'] = context['trino']
        return template.render(context)

DEFAULT_PROCESSORS: dict[str, Callable] = {'presto': PrestoTemplateProcessor, 'hive': HiveTemplateProcessor, 'spark': SparkTemplateProcessor, 'trino': TrinoTemplateProcessor}

@lru_cache(maxsize=LRU_CACHE_MAX_SIZE)
def get_template_processors() -> dict[str, Callable]:
    processors = current_app.config.get('CUSTOM_TEMPLATE_PROCESSORS', {})
    for engine, processor in DEFAULT_PROCESSORS.items():
        if engine not in processors:
            processors[engine] = processor
    return processors

def get_template_processor(database: Database, table: Optional[SqlaTable] = None, query: Optional[Query] = None, **kwargs: Any) -> Callable:
    if feature_flag_manager.is_feature_enabled('ENABLE_TEMPLATE_PROCESSING'):
        template_processor = get_template_processors().get(database.backend, JinjaTemplateProcessor)
    else:
        template_processor = NoOpTemplateProcessor
    return template_processor(database=database, table=table, query=query, **kwargs)

def dataset_macro(dataset_id: int, include_metrics: bool = False, columns: Optional[list] = None, from_dttm: Optional[str] = None, to_dttm: Optional[str] = None) -> str:
    """
    Given a dataset ID, return the SQL that represents it.

    The generated SQL includes all columns (including computed) by default. Optionally
    the user can also request metrics to be included, and columns to group by.

    The from_dttm and to_dttm parameters are filled in from filter values in explore
    views, and we take them to make those properties available to jinja templates in
    the underlying dataset.
    """
    from superset.daos.dataset