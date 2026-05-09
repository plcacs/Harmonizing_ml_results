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
from superset.connectors.sqla.models import SqlaTable
from superset.models.core import Database
from superset.models.sql_lab import Query

NONE_TYPE: str = type(None).__name__
ALLOWED_TYPES: tuple[str, ...] = (NONE_TYPE, 'bool', 'str', 'unicode', 'int', 'long', 'float', 'list', 'dict', 'tuple', 'set')
COLLECTION_TYPES: tuple[str, ...] = ('list', 'dict', 'tuple', 'set')

@lru_cache(maxsize=LRU_CACHE_MAX_SIZE)
def context_addons() -> dict[str, Any]:
    return current_app.config.get('JINJA_CONTEXT_ADDONS', {})

class Filter(TypedDict):
    pass

@dataclass
class TimeFilter:
    """Container for temporal filter."""

class ExtraCache:
    """Dummy class that exposes a method used to store additional values used in calculation of query object cache keys."""

    def __init__(self, extra_cache_keys: Optional[list[str]] = None, applied_filters: Optional[list[Filter]] = None, removed_filters: Optional[list[str]] = None, database: Optional[Database] = None, dialect: Optional[Dialect] = None, table: Optional[Table] = None) -> None:
        self.extra_cache_keys: list[str] = extra_cache_keys
        self.applied_filters: list[Filter] = applied_filters if applied_filters is not None else []
        self.removed_filters: list[str] = removed_filters if removed_filters is not None else []
        self.database: Database = database
        self.dialect: Dialect = dialect
        self.table: Table = table

    def current_user_id(self, add_to_cache_keys: bool = True) -> Optional[int]:
        """Return the user ID of the user who is currently logged in."""
        if (user_id := get_user_id()):
            if add_to_cache_keys:
                self.cache_key_wrapper(user_id)
            return user_id
        return None

    def current_username(self, add_to_cache_keys: bool = True) -> Optional[str]:
        """Return the username of the user who is currently logged in."""
        if (username := get_username()):
            if add_to_cache_keys:
                self.cache_key_wrapper(username)
            return username
        return None

    def current_user_email(self, add_to_cache_keys: bool = True) -> Optional[str]:
        """Return the email address of the user who is currently logged in."""
        if (email_address := get_user_email()):
            if add_to_cache_keys:
                self.cache_key_wrapper(email_address)
            return email_address
        return None

    def cache_key_wrapper(self, key: Any) -> Any:
        """Adds values to a list that is added to the query object used for calculating a cache key."""
        if self.extra_cache_keys is not None:
            self.extra_cache_keys.append(key)
        return key

    def url_param(self, param: str, default: Optional[Any] = None, add_to_cache_keys: bool = True, escape_result: bool = True) -> Optional[str]:
        """Read a url or post parameter and use it in your SQL Lab query."""
        from superset.views.utils import get_form_data
        form_data, _ = get_form_data()
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

    def filter_values(self, column: str, default: Optional[Any] = None, remove_filter: bool = False) -> list[Any]:
        """Gets a values for a particular filter as a list."""
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

    def get_filters(self, column: str, remove_filter: bool = False) -> list[Filter]:
        """Get the filters applied to the given column."""
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

    def get_time_filter(self, column: Optional[str] = None, default: Optional[Any] = None, target_type: Optional[str] = None, strftime: Optional[str] = None, remove_filter: bool = False) -> TimeFilter:
        """Get the time filter with appropriate formatting, either for a specific column, or whichever time range is being emitted from a dashboard."""
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
        def _format_dttm(dttm: Optional[datetime]) -> Optional[str]:
            if strftime and dttm:
                return dttm.strftime(strftime)
            return self.database.db_engine_spec.convert_dttm(target_type or '', dttm) if self.database and dttm else None
        return TimeFilter(from_expr=_format_dttm(from_expr), to_expr=_format_dttm(to_expr), time_range=time_range)

def safe_proxy(func: Callable[..., Any], *args: tuple[Any, ...], **kwargs: dict[str, Any]) -> Any:
    """A decorator that validates the return type of a function."""
    return_value = func(*args, **kwargs)
    value_type = type(return_value).__name__
    if value_type not in ALLOWED_TYPES:
        raise SupersetTemplateException(_('Unsafe return type for function %(func)s: %(value_type)s', func=func.__name__, value_type=value_type